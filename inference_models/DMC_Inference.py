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

from inference_models.base.DMC_Base import (
    EmbeddingFCGRU,
)
from inference_models.base.base import BaseInferenceModel
from model_engine.util import per_task_param_loader
from model_engine.engine import BatchModelEngine

import copy

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
        self.curr_batch_size = None

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

    def reset(self, batch_size: int=1) -> None:
        """
        Reset the interal model
        """
        self.curr_batch_size = batch_size
        self.nn.zero_grad()
        self.model.reset()
        self.hn = self.nn.get_init_state(batch_size)

    def predict(self, data: np.ndarray, dates: np.ndarray, cultivars: np.ndarray) -> np.ndarray:
        """
        Docstring for predict
        Predicts crop state given that the weather is assumed to be the true weather (advances the crop model)

        :param data: Daily weather data of shape [num_cultivars, num_days, num_features] (assumed to be true weather)
        :param dates: Array of dates of shape [num_days]
        :param cultivars: Array of cultivars as ints or strings to predict for
        :return: Array of phenology predictions of shape [num_cultivars, num_days, 1]
        """

        # Validate input shapes and modify as needed
        data, dates, cultivars = self.validate_input(data, dates, cultivars)

        # Process data into form usable by NN
        nn_data, dates, nn_cultivars = self.process_nn_data(data, dates, cultivars)

        model_output = self.model.get_output()[:self.curr_batch_size]
        output_tens = torch.empty(size=(data.shape[0], data.shape[1], len(self.output_vars))).to(self.device)

        # Go through every day in the data array and predict parameters + model output
        for i in range(nn_data.shape[1]):
            with torch.no_grad(): 
                params_predict, self.hn = self.nn(
                    torch.cat((model_output.view(model_output.shape[0], -1).detach(), nn_data[:, i]), dim=-1),
                    hn=self.hn,
                    cultivars=nn_cultivars,
                )
                params_predict = self.param_cast(params_predict)
                self.model.set_model_params(params_predict, self.params)
                model_output = self.model.run(weather_data=data[:, i], dates=dates[i])[:self.curr_batch_size]
                output_tens[:, i] = model_output
  
        return output_tens.detach().cpu().numpy(), self.hn
    
    def forecast(self, data: np.ndarray, dates: np.ndarray, cultivars: np.ndarray) -> np.ndarray:
        """
        Docstring for forecast
        Forecasts crop state given a weather forecast. Does not advance crop model, resets to current model state
        after performing forecast rollout.

        :param data: Daily weather data of shape [num_cultivars, num_days, num_features] (assumed to be true weather)
        :param dates: Array of dates of shape [num_days]
        :param cultivars: Array of cultivars as ints or strings to predict for
        :return: Array of phenology predictions of shape [num_cultivars, num_days, 1]
        """

        # Validate input shapes and modify as needed
        data, dates, cultivars = self.validate_input(data, dates, cultivars)

        # Process data into form usable by NN
        nn_data, dates, nn_cultivars = self.process_nn_data(data, dates, cultivars)

        model_output = self.model.get_output()[:self.curr_batch_size]
        output_tens = torch.empty(size=(data.shape[0], data.shape[1], len(self.output_vars))).to(self.device)

        # Save current state of model 
        curr_model_state = copy.deepcopy(self.model.get_state())
        curr_hn = self.hn.clone()

        # Go through every day in the data array and predict parameters + model output
        for i in range(nn_data.shape[1]):
            with torch.no_grad():
                params_predict, curr_hn = self.nn(
                    torch.cat((model_output.view(model_output.shape[0], -1).detach(), nn_data[:, i]), dim=-1),
                    cultivars=nn_cultivars,
                    hn=curr_hn
                )

                params_predict = self.param_cast(params_predict)
                self.model.set_model_params(params_predict, self.params)
                model_output = self.model.run(weather_data=data[:, i], dates=dates[i])
                output_tens[:, i] = model_output

        # Reset model internal state
        self.model.set_state(curr_model_state)

        return output_tens.detach().cpu().numpy()

class InferenceNoObsParamRNN(BaseInferenceRNN):

    def __init__(self, config: DictConfig, data: list[pd.DataFrame]) -> None:

        super(InferenceNoObsParamRNN, self).__init__(config, data)

        self.model = BatchModelEngine(
            num_models=self.batch_size,
            config=config.PConfig,
            device=self.device,
        )

        self.task_params = per_task_param_loader(config, self.params).to(self.device)

