"""
DMC_Inference.py

Contains classes for performing inference on RNN model

Written by Will Solow, 2025
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from train_algs.base.DMC_Base import (
    EmbeddingFCGRU,
)
from train_algs.base.base import BaseInferenceModel
from model_engine.util import per_task_param_loader
from model_engine.engine import BatchModelEngine


class BaseInferenceRNN(BaseInferenceModel):

    def __init__(self, config: DictConfig, fpath: str, pt_file_name: str = "rnn_model.pt") -> None:

        super().__init__(config)

        self.nn = BaseInferenceRNN.make_rnn(self, config).to(self.device)
        (
            self.nn.rnn.flatten_parameters()
            if hasattr(self.nn, "rnn") and hasattr(self.nn.rnn, "flatten_parameters")
            else None
        )

        self.load_model(fpath, pt_file_name)
        self.drange = torch.load(f"{fpath}/model_drange.pt").to(self.device)

    @staticmethod
    def make_rnn(model: nn.Module, config: DictConfig) -> nn.Module:
        """Make the RNN"""

        # Create RNN Model
        if config.DConfig.arch == "EmbedFCGRU":
            nn = EmbeddingFCGRU(config, model)
        else:
            raise Exception(f"Unrecognized Model Architecture `{config.DConfig.arch}`")

        return nn


class InferenceParamRNN(BaseInferenceRNN):

    def __init__(self, config: DictConfig, fpath: str, pt_file_name="rnn_model.pt") -> None:

        super(InferenceParamRNN, self).__init__(config, fpath, pt_file_name=pt_file_name)

        self.model = BatchModelEngine(
            num_models=self.batch_size,
            config=config.PConfig,
            device=self.device,
        )

        self.task_params = per_task_param_loader(config, self.params).to(self.device)

    def reset(self) -> None:
        """
        Reset the interal model
        """

        self.nn.zero_grad()
        self.model.reset()

    def infer(self, data: np.ndarray, dates: np.ndarray, cultivars: np.ndarray) -> np.ndarray:
        """
        Docstring for infer

        :param data: Daily weather data of shape [num_cultivars, num_days, num_features]
        :param dates: Array of dates of shape [num_days]
        :param cultivars: Array of cultivars to predict for
        :return: Array of phenology predictions of shape [num_cultivars, num_days, 1]
        """

        # Validate input shapes and modify as needed
        data, dates, cultivars = self.validate_input(data, dates, cultivars)

        # Process data into form usable by NN
        nn_data, dates, nn_cultivars = self.process_nn_data(data, dates, cultivars)

        model_output = self.model.get_output()[: len(cultivars)]
        output_tens = torch.empty(size=(data.shape[0], data.shape[1], len(self.output_vars))).to(self.device)

        # Go through every day in the data array and predict parameters + model output
        for i in range(nn_data.shape[1]):
            params_predict, _ = self.nn(
                torch.cat((model_output.view(model_output.shape[0], -1).detach(), nn_data[:, i]), dim=-1),
                cultivars=nn_cultivars,
            )

            params_predict = self.param_cast(params_predict)
            self.model.set_model_params(params_predict, self.params)
            output = self.model.run(weather_data=data[:, i], dates=dates[:, i])
            output_tens[:, i] = output

        return output_tens.detach().cpu().numpy()

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

        # Set up
        hn_cn = self.nn.get_init_state(batch_size=data.shape[0]) if hasattr(self.nn, "get_init_state") else None
        batch_params = (
            self.task_params[cultivars.to(torch.long).squeeze()] if cultivars is not None else self.task_params[0]
        )
        self.model.set_model_params(batch_params, self.params)
        output = self.model.reset()[:b_size]

        # Run through entirety of time series predicting params
        for i in range(dlen):
            params_predict, hn_cn = self.nn(
                torch.cat((output.view(output.shape[0], -1).detach(), data[:, i]), dim=-1),
                hn=hn_cn,
                cultivars=cultivars,
            )

            params_predict = self.param_cast(params_predict)
            self.model.set_model_params(params_predict, self.params)
            output = self.model.run(dates=dates[:, i], cultivars=cultivars)

            output_tens[:, i] = output
            param_tens[:, i] = params_predict

        return output_tens, param_tens, None


class InferenceNoObsParamRNN(BaseInferenceRNN):

    def __init__(self, config: DictConfig, data: list[pd.DataFrame]) -> None:

        super(InferenceNoObsParamRNN, self).__init__(config, data)

        self.model = BatchModelEngine(
            num_models=self.batch_size,
            config=config.PConfig,
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
        output = self.model.reset()[:b_size]
        # Run through entirety of time series predicting parameters for physical model at each step
        for i in range(dlen):

            params_predict, hn_cn = self.nn(data[:, i], hn_cn, cultivars=cultivars)
            params_predict = self.param_cast(params_predict)
            self.model.set_model_params(params_predict, self.params)

            output = self.model.run(dates=dates[:, i], cultivars=cultivars)

            output_tens[:, i] = output
            param_tens[:, i] = params_predict

        return output_tens, param_tens, None
