"""
base.py

Base nn.Module class

Written by Will Solow, 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from omegaconf import DictConfig
import pandas as pd
import numpy as np
import random
from train_algs.base.process_data import process_data
from model_engine.util import PHENOLOGY_INT


class BaseModel(nn.Module):

    def __init__(self, config: DictConfig, data: list[pd.DataFrame]) -> None:

        super(BaseModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        np.random.seed(config.seed)
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        self.config = config
        self.target_mask = np.nan

        process_data(self, data)

    def make_optimizer(self, model: nn.Module) -> None:

        self.learning_rate = self.config.DConfig.learning_rate
        self.batch_size = self.config.DConfig.batch_size
        self.epochs = self.config.DConfig.epochs

        self.loss_func = nn.MSELoss(reduction="none")

        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.config.DConfig.lr_factor,
            patience=10,
            cooldown=10,
            threshold=1e-6,
        )

    def setup_storage(self, b_size: int, dlen: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Set up storage for forward pass
        """
        output_dim = len(self.output_vars)

        output_tens = torch.empty(size=(b_size, dlen, output_dim)).to(self.device)
        model_output_tens = torch.empty(size=(b_size, dlen, output_dim)).to(self.device)
        param_tens = torch.empty(size=(b_size, dlen, len(self.params))).to(self.device)

        return output_tens, param_tens, model_output_tens

    def handle_data(self, data: torch.Tensor, dates: np.ndarray, cultivars: torch.Tensor):
        """
        Handle unbatched data
        """
        if data.dim() == 2:
            data = data.unsqueeze(0)
            dates = dates[np.newaxis, :]
        if cultivars is not None:
            if cultivars.dim() == 1:
                cultivars = cultivars.unsqueeze(0)

        b_size = data.shape[0]
        dlen = data.shape[1]

        return data, dates, cultivars, b_size, dlen

    def forward(
        self, data: torch.Tensor, dates: torch.Tensor, cultivars: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass on the year of data to output parameters
        Assumes unbatched
        """
        raise NotImplementedError

    def optimize(self) -> None:
        """Optimize function to be implemented in base class"""
        raise NotImplementedError

    def save_model(self, path: str, name: str = "rnn_model.pt") -> None:
        """
        Save model
        """
        torch.save(self.nn.state_dict(), f"{path}/{name}")

    def load_model(self, path: str, name: str = "rnn_model.pt") -> None:
        """
        Load Model
        """
        self.nn.load_state_dict(torch.load(f"{path}/{name}", weights_only=True, map_location=self.device), strict=False)

        try:
            self.nn.load_state_dict(
                torch.load(f"{path}/{name}", weights_only=True, map_location=self.device), strict=False
            )
        except:
            print("Unable to load NN model..")

    def param_cast(self, params: torch.Tensor, prev_params: torch.Tensor = None) -> torch.Tensor:
        """
        Performs TanH activiation on parameters to cast to range
        """
        # Cast to range [0,2] from tanh activation and cast to actual parameter range

        if self.config.param_scale != None:
            params_predict = torch.tanh(params)
            params_predict = 0 + params_predict * (self.params_range[:, 1] - self.params_range[:, 0]) / 2
            params_predict = params_predict * self.config.param_scale
            params_predict = prev_params + params_predict
            params_predict = torch.clamp(params_predict, self.params_range[:, 0], self.params_range[:, 1])
        else:
            params_predict = torch.tanh(params) + 1
            params_predict = (
                self.params_range[:, 0] + params_predict * (self.params_range[:, 1] - self.params_range[:, 0]) / 2
            )

        return params_predict

    def get_input_dim(self, config: DictConfig):
        """
        Get the input dimension of the model
        """
        extra_feat = (
            0
            if config.DConfig.type in ["Deep", "Hybrid", "NoObsParam", "Residual", "WindowParam"]
            else len(config.PConfig.output_vars)
        )
        extra_feat = (
            extra_feat + 1 if "DAY" in config.PConfig.input_vars else extra_feat
        )  # +1 is extra term for date embedding

        return extra_feat + len(config.PConfig.input_vars)

    def get_output_dim(self, config: DictConfig):
        """
        Get the output dimension of the model
        """
        if config.DConfig.type in ["Param", "NoObsParam"]:
            return len(config.params)
        else:
            return len(config.PConfig.output_vars)

    def get_ft_input_dim(self, config: DictConfig):
        """
        Get the input dimension for finetuning models
        """
        return len(config.PConfig.output_vars)

    def compute_model_error(self, day, output, target, dates, cultivars):
        """
        Compute the error in the model by rolling it out with the current params
        until all phenologies are equal, then compute this error
        """
        errors = torch.zeros((output.size(0), 1)).to(self.device)

        model_output = output.detach().clone()
        curr_model_state = self.model.get_state()

        too_soon_mask = (model_output > target[:, day]) & (~torch.isnan(target[:, day]))
        too_late_mask = (model_output < target[:, day]) & (~torch.isnan(target[:, day]))
        new_late_mask = too_late_mask.clone()

        # Here, we find all the times the model under estimated the prediction for the onset of a stage (predict too soon)
        args = torch.argwhere(
            (
                torch.broadcast_to(too_soon_mask.unsqueeze(1), target.shape) & (target == model_output.unsqueeze(1))
            ).squeeze()
        )
        for b in range(len(dates)):
            inds = args[args[:, 0] == b]
            if len(inds) != 0:
                errors[b] = day - inds[0][1]

        # Here, we find all the times the over estimated the prediction for the onset of the state (predict too late)
        # We must roll out the model to find this. We assume the parameters don't change and that
        # We know the weather perfectly
        i = day + 1
        while (model_output != target[:, day])[new_late_mask].any() and i < dates.shape[1]:
            model_output = self.model.run(dates=dates[:, i], cultivars=cultivars)
            i += 1
            new_late_mask = (model_output < target[:, day]) & (~torch.isnan(target[:, day])) & too_late_mask
            errors[new_late_mask] += (model_output != target[:, day])[new_late_mask]
        self.model.set_state(curr_model_state)

        return errors
