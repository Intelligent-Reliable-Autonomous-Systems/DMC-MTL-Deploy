"""
base.py

Base nn.Module class

Written by Will Solow, 2025
"""

import torch
import torch.nn as nn

from omegaconf import DictConfig
import numpy as np
import random
from inference_models.base.process_data import process_data_inference, date_to_cyclic, normalize
from model_engine.util import CULTIVARS
from inference_models.base.util import assert_yyyy_mm_dd


class BaseInferenceModel(nn.Module):

    def __init__(self, config: DictConfig) -> None:

        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        np.random.seed(config.seed)
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        self.config = config

        process_data_inference(self)

        self.batch_size = self.config.DConfig.batch_size

    def validate_input(
        self, data: np.ndarray, dates: np.ndarray, cultivars: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        assert cultivars.ndim == 1, "`cultivars` must be 1-dimensional"
        assert dates.ndim == 1, "`dates` must be 1 dimensional"
        assert self.curr_batch_size is not None, "Current batch size must not be None. Call `model.reset()` to set."

        assert np.issubdtype(data.dtype, np.number), f"`data` must be of type `number` but if of type `{data.dtype}`"
        assert np.issubdtype(cultivars.dtype, np.number) or np.issubdtype(cultivars.dtype, np.str_), f"`cultivars` must be of type `number` or `str` but if of type `{cultivars.dtype}`"
        assert np.issubdtype(dates.dtype, np.datetime64) or np.issubdtype(dates.dtype, np.str_), f"`dates` must be of type `str` or `datetime64` but if of type `{dates.dtype}`"

        dates = assert_yyyy_mm_dd(dates)

        # Handles 
        if np.issubdtype(cultivars.dtype, np.str_):
            index_map = {v: i for i, v in enumerate(CULTIVARS[self.config.DataConfig.dtype])}
            invalid = [x for x in cultivars if x not in index_map]
            if invalid:
                raise ValueError(f"Invalid cultivars: {invalid}")
            cultivars = np.array([index_map[x] for x in cultivars])

        # Data check that data sizes are compatible

        if data.ndim == 1:
            data = np.expand_dims(data, axis=(0, 1))
        if data.ndim == 2:
            if data.shape[0] == dates.shape[0]:
                data = np.expand_dims(data, axis=0)
            elif data.shape[0] == cultivars.shape[0]:
                data = np.expand_dims(data, axis=1)
            else:
                raise Exception(
                    f"Incompatible dimensions between data: {data.shape}, dates: {dates.shape} and cultivars: {cultivars.shape}."
                )
        assert data.ndim == 3, "Data is not 3-dimensional"
        assert (
            data.shape[0] == cultivars.shape[0] and data.shape[1] == dates.shape[0]
        ), f"Shape mismatch betwen data: {data.shape}, dates: {dates.shape} and cultivars: {cultivars.shape}."
        assert data.shape[0] == self.curr_batch_size, f"Dimension 0 of `data` must be equal to `curr_batch_size`. Expected {self.curr_batch_size} but got {data.shape[0]}"

        return data, dates, cultivars

    def process_nn_data(
        self, data: np.ndarray, dates: np.ndarray, cultivars: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process the data and dates so that it is usable by the neural network
        """
        if "DAY" in self.config.PConfig.input_vars:
            cyclic_dates = date_to_cyclic(dates)
            nn_data = torch.tensor(np.concatenate((cyclic_dates[np.newaxis, :], data), axis=-1)).to(self.device)
        else:
            nn_data = torch.tensor(data).to(self.device)
        nn_data = normalize(nn_data, self.drange).to(torch.float32)

        try: 
            nn_cultivars = torch.tensor(cultivars).unsqueeze(0).to(self.device).to(torch.int)
        except:
            raise Exception(f"Unable to convert `cultivars`:{cultivars} into integer tensor.")

        return nn_data, dates[:, np.newaxis], nn_cultivars

    def load_model(self, path: str, name: str = "rnn_model.pt") -> None:
        """
        Load Model
        """
        self.nn.load_state_dict(torch.load(f"{path}/{name}", weights_only=True, map_location=self.device), strict=False)

    def param_cast(self, params: torch.Tensor) -> torch.Tensor:
        """
        Performs TanH activiation on parameters to cast to range
        """
        # Cast to range [0,2] from tanh activation and cast to actual parameter range
        params_predict = torch.tanh(params) + 1
        params_predict = (
            self.params_range[:, 0] + params_predict * (self.params_range[:, 1] - self.params_range[:, 0]) / 2
        )

        return params_predict

    def forward(
        self, data: torch.Tensor, dates: torch.Tensor, cultivars: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass on the year of data to output parameters
        Assumes unbatched
        """
        raise NotImplementedError

    def save_model(self, path: str, name: str = "rnn_model.pt") -> None:
        """
        Save model
        """
        torch.save(self.nn.state_dict(), f"{path}/{name}")

    def param_cast(self, params: torch.Tensor, prev_params: torch.Tensor = None) -> torch.Tensor:
        """
        Performs TanH activiation on parameters to cast to range
        """
        # Cast to range [0,2] from tanh activation and cast to actual parameter range

        params_predict = torch.tanh(params) + 1
        params_predict = (
            self.params_range[:, 0] + params_predict * (self.params_range[:, 1] - self.params_range[:, 0]) / 2
        )

        return params_predict

    def get_input_dim(self, config: DictConfig):
        """
        Get the input dimension of the model
        """
        extra_feat = 0 if config.DConfig.type in ["NoObsParam"] else len(config.PConfig.output_vars)
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
