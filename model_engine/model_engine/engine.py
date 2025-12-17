"""
engine.py

Contains multiple engines that wrap around a model class for a unified
API.

BaseEngine - base engine class that all others inherit
SingleModelEngine - Assumes that output from model is not batched
MultiModelEngine - Assumes that output from model is not batched but runs multiple
models simultaneously
BatchModelEngine - Assumes that output from model is batched, only compatible with
batch models


Written by Will Solow, 2025
"""

import datetime
from datetime import date
import os
import numpy as np
import torch
from traitlets_pcse import Instance, HasTraits
from omegaconf import DictConfig

from model_engine.util import param_loader, get_models
from model_engine.inputs.input_providers import DFTensorWeatherDataContainer
from model_engine.models.states_rates import VariableKiosk


class BaseEngine(HasTraits):
    """
    Base Wrapper class for models
    """

    def __init__(
        self,
        config: DictConfig = None,
        device: torch.device = "cpu",
    ) -> None:
        self.device = device
        self.config = config
        self.start_date = np.datetime64(config.start_date)
        self.day = self.start_date
        self.kiosk = VariableKiosk()

        self.output_vars = self.config.output_vars
        self.input_vars = self.config.input_vars
        self.weather_input_vars = [x for x in self.input_vars if x != "DAY"]

        self.model_constr = get_models(f"{os.path.dirname(os.path.abspath(__file__))}/models")[config.model]

    def run(self, dates: datetime.date = None) -> torch.Tensor:
        """
        Advances the system state with given number of days
        """
        self._run(date=dates)

        return self.get_output()

    def _run(self, date: datetime.date = None) -> None:
        """
        Helper function for run. Implemented by subclasses
        """
        raise NotImplementedError

    def get_state_rates_names(self) -> list[str]:
        """
        Get names of valid states and ratse
        """
        return self.model.get_state_rates_names()

    def get_output(self) -> torch.Tensor:
        """
        Get the output of a model
        """
        raise NotImplementedError


class BatchModelEngine(BaseEngine):
    """
    Wrapper class for the BatchModelEngine around Batch Models
    Model must be a tensormodel model.
    This is the best option for wrapping models if a batch model is
    available
    """

    days = Instance(np.ndarray)

    def __init__(
        self,
        num_models: int = 1,
        config: DictConfig = None,
        device: torch.device = "cpu",
    ) -> None:

        super().__init__(config, device)
        self.num_models = num_models
        self.model = self.model_constr(
            self.start_date,
            self.kiosk,
            param_loader(self.config),
            self.device,
            num_models=self.num_models,
        )

    def reset(self) -> torch.Tensor:
        """
        Reset the model
        """

        self.day = self.start_date

        self.model.reset(self.day)

        return self.get_output()

    def run(
        self, weather_data: torch.Tensor = None, dates: np.ndarray = None
    ) -> torch.Tensor:
        """
        Advances the system state with given number of days
        """

        self._run(weather_data=weather_data, dates=dates)

        return self.get_output()[: len(dates)] if dates is not None else self.get_output()

    def _run(
        self,
        weather_data: torch.Tensor = None,
        dates: datetime.date = None,
        delt: float = 1,
    ) -> torch.Tensor:
        """
        Make one time step of the simulation.
        """
        n_pad = self.num_models - weather_data.shape[0]
        days = (
            np.pad(
                dates,
                (0, n_pad),
                mode="constant",
                constant_values=dates[-1],
            )
            if n_pad > 0
            else dates
        )
        if n_pad > 0:
            weather_data = np.vstack([weather_data, np.tile(weather_data[-1:], (n_pad, 1))])
            
        weather = dict(zip(self.weather_input_vars, weather_data.transpose()))
        drv = DFTensorWeatherDataContainer(device=self.device, **weather)

        self.calc_rates(days, drv)
        self.integrate(days, delt)

    def calc_rates(self, day: date, drv: DFTensorWeatherDataContainer) -> None:
        """
        Calculate the rates for computing rate of state change
        """
        self.model.calc_rates(day, drv)

    def integrate(self, day: date, delt: float) -> None:
        """
        Integrate rates with states based on time change (delta)
        """
        self.model.integrate(day, delt)

    def set_model_params(self, new_params: torch.Tensor, param_list: list) -> None:
        """
        Set the model parameters
        """
        if new_params.ndim < 2:
            new_params = new_params.unsqueeze(0)
        if new_params.shape[0] < self.num_models:
            bsize = new_params.shape[0]
            new_params = torch.nn.functional.pad(new_params, (0, 0, 0, self.num_models - new_params.shape[0]), value=0)
            new_params[bsize:] = new_params[0]
        self.model.set_model_params(dict(zip(param_list, torch.split(new_params, 1, dim=-1))))

    def get_output(self, output_vars: list = None) -> torch.Tensor:
        """
        Get the observable output of the model
        """
        return self.model.get_output(va=self.output_vars if output_vars == None else output_vars)

    def get_params(self) -> dict:
        """
        Get the parameter dictionary
        """
        return self.model.get_params()

    def get_state(self, i: int = None) -> list[dict, torch.Tensor]:
        """
        Get the state of the model
        """
        state = self.model.get_state_rates()
        extra_vars = state[0]

        return [extra_vars, torch.stack(state[1:], dim=-1).to(self.device)]

    def set_state(self, state: list[dict : torch.Tensor], i: int = None) -> torch.Tensor:
        """
        Set the state of the model
        """
        rep_factor = (self.num_models + state[1].size(0) - 1) // state[1].size(0)
        extra_vars = state[0]

        state_rates = state[1].repeat(rep_factor, 1)[: self.num_models].T
        self.model.set_state_rates([extra_vars, state_rates])

        return self.get_output()
