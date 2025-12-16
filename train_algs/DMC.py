"""
ParamRNN.py

Contains classes for running Param RNN models. We assume that
the model predicts parameters for a physical model.

Written by Will Solow, 2025
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from omegaconf import OmegaConf, DictConfig

from train_algs.base.DMC_Base import (
    EmbeddingFCGRU,
)
from train_algs.base.base import BaseModel
from model_engine.util import per_task_param_loader
from model_engine.engine import BatchModelEngine

from train_algs.base.util import (
    setup_logging,
    cumulative_error,
    get_grad_norm,
    log_training,
)


class BaseRNN(BaseModel):

    def __init__(self, config: DictConfig, data: list[pd.DataFrame]) -> None:

        super().__init__(config, data)

        self.nn = BaseRNN.make_rnn(self, config).to(self.device)
        (
            self.nn.rnn.flatten_parameters()
            if hasattr(self.nn, "rnn") and hasattr(self.nn.rnn, "flatten_parameters")
            else None
        )
        self.make_optimizer(self.nn)

    @staticmethod
    def make_rnn(model: nn.Module, config: DictConfig) -> nn.Module:
        """Make the RNN"""

        # Create RNN Model
        if config.DConfig.arch == "EmbedFCGRU":
            nn = EmbeddingFCGRU(config, model)
        else:
            raise Exception(f"Unrecognized Model Architecture `{config.DConfig.arch}`")

        return nn

    def optimize(self) -> None:

        writer, run_name, log_path = setup_logging(self.config)
        os.makedirs(log_path, exist_ok=True)

        with open(f"{log_path}/config.yaml", "w", encoding="utf-8") as fp:
            OmegaConf.save(config=self.config, f=fp.name)
        fp.close()

        self.best_cum_rmse = float("inf")
        self.best_eval_loss = float("inf")
        self.best_rmse = torch.zeros(size=(4,)).to(self.device)

        for param in self.nn.parameters():
            param.requires_grad = True
        self.nn.train()

        train_name = "train"
        test_name = "test"
        for epoch in range(self.epochs):

            train_loss = 0
            grad = 0
            inds = np.arange(len(self.data[train_name]))
            np.random.shuffle(inds)
            data_shuffled = self.data[train_name][inds]
            val_shuffled = self.val[train_name][inds]
            dates_shuffled = self.dates[train_name][inds]
            cultivars_shuffled = self.cultivars[train_name][inds] if self.cultivars is not None else None
            train_avg = torch.zeros(size=(4,)).to(self.device)

            # Training
            for i in range(0, len(self.data[train_name]), self.batch_size):
                self.optimizer.zero_grad()

                batch_data = data_shuffled[i : i + self.batch_size]
                batch_dates = dates_shuffled[i : i + self.batch_size]
                batch_cultivars = (
                    cultivars_shuffled[i : i + self.batch_size] if cultivars_shuffled is not None else None
                )
                target = val_shuffled[i : i + self.batch_size]
                output, _, model_output = self.forward(
                    batch_data,
                    batch_dates,
                    cultivars=batch_cultivars,
                )

                loss = self.loss_func(output, target.nan_to_num(nan=0.0))

                mask = ~torch.isnan(target)
                loss = (loss * mask).sum() / mask.sum()
                loss.backward()

                self.optimizer.step()
                train_loss += loss.item()
                grad += get_grad_norm(self.nn)

                avg_ = cumulative_error(target, output, mask, device=self.device)
                train_avg[:3] += avg_[1:-1]
                train_avg[-1] += torch.sum(avg_[1:-1]).to(self.device)

            # Evaluation
            eval_loss = 0
            eval_avg = torch.zeros(size=(4,)).to(self.device)

            for j in range(0, len(self.data[test_name]), self.batch_size):
                self.optimizer.zero_grad()
                batch_data = self.data[test_name][j : j + self.batch_size]
                batch_dates = self.dates[test_name][j : j + self.batch_size]
                batch_cultivars = (
                    self.cultivars[test_name][j : j + self.batch_size] if self.cultivars is not None else None
                )
                eval_target = self.val[test_name][j : j + self.batch_size]
                eval_output, _, eval_model_output = self.forward(
                    batch_data,
                    batch_dates,
                    cultivars=batch_cultivars,
                )

                eval_loss = self.loss_func(eval_output, eval_target.nan_to_num(nan=0.0))
                eval_mask = ~torch.isnan(eval_target)
                eval_loss = (eval_loss * eval_mask).sum() / eval_mask.sum()
                eval_loss += eval_loss.item()

                avg_ = cumulative_error(eval_target, eval_output, eval_mask, device=self.device)
                eval_avg[:3] += avg_[1:-1]
                eval_avg[-1] += torch.sum(avg_[1:-1]).to(self.device)

            log_training(
                self,
                writer,
                log_path,
                epoch,
                train_loss,
                eval_loss,
                train_avg,
                eval_avg,
                grad,
            )

        self.scheduler.step(float(train_loss))


class ParamRNN(BaseRNN):

    def __init__(self, config: DictConfig, data: list[pd.DataFrame]) -> None:

        super(ParamRNN, self).__init__(config, data)

        self.model = BatchModelEngine(
            num_models=self.batch_size,
            config=config.PConfig,
            inputprovider=self.input_data,
            device=self.device,
        )

        self.task_params = per_task_param_loader(config, self.params).to(self.device)

    def forward(
        self,
        data: torch.Tensor,
        dates: np.ndarray,
        cultivars: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass call for integrated gradients attribution sequence batch data
        """
        self.nn.zero_grad()

        data, dates, cultivars, b_size, dlen = self.handle_data(data, dates, cultivars)

        output_tens, param_tens, _ = self.setup_storage(b_size, dlen)

        # Hidden state
        hn_cn = self.nn.get_init_state(batch_size=data.shape[0]) if hasattr(self.nn, "get_init_state") else None
        batch_params = (
            self.task_params[cultivars.to(torch.long).squeeze()] if cultivars is not None else self.task_params[0]
        )
        self.model.set_model_params(batch_params, self.params)
        output = self.model.reset(b_size)

        # Run through entirety of time series predicting params
        for i in range(dlen):
            params_predict, hn_cn = self.nn(
                torch.cat((output.view(output.shape[0], -1).detach(), data[:, i]), dim=-1),
                hn=hn_cn,
                cultivars=cultivars,
            )

            params_predict = self.param_cast(params_predict, prev_params=batch_params)
            batch_params = params_predict
            self.model.set_model_params(params_predict, self.params)
            output = self.model.run(dates=dates[:, i], cultivars=cultivars)

            output_tens[:, i] = output
            param_tens[:, i] = params_predict

        return output_tens, param_tens, None

class NoObsParamRNN(BaseRNN):

    def __init__(self, config: DictConfig, data: list[pd.DataFrame]) -> None:

        super(NoObsParamRNN, self).__init__(config, data)

        self.model =BatchModelEngine(
            num_models=self.batch_size,
            config=config.PConfig,
            inputprovider=self.input_data,
            device=self.device,
        )

        self.task_params = per_task_param_loader(config, self.params).to(self.device)

    def forward(
        self, data: torch.Tensor = None, dates: np.ndarray = None, cultivars: torch.Tensor = None, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass call for integrated gradients attribution sequence batch data
        """
        data, dates, cultivars, b_size, dlen = self.handle_data(data, dates, cultivars)

        output_tens, param_tens, _ = self.setup_storage(b_size, dlen)

        self.nn.zero_grad()
        hn_cn = self.nn.get_init_state(batch_size=b_size)
        output = self.model.reset(b_size)
        # Run through entirety of time series predicting parameters for physical model at each step
        for i in range(dlen):

            params_predict, hn_cn = self.nn(data[:, i], hn_cn, cultivars=cultivars)
            params_predict = self.param_cast(params_predict)
            self.model.set_model_params(params_predict, self.params)

            output = self.model.run(dates=dates[:, i], cultivars=cultivars)

            output_tens[:, i] = output
            param_tens[:, i] = params_predict

        return output_tens, param_tens, None
