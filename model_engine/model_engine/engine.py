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
import torch.nn.functional as F
from traitlets_pcse import Instance, HasTraits
import pandas as pd
from omegaconf import DictConfig

from model_engine.util import param_loader, get_models
from model_engine.inputs.nasapower import NASAPowerWeatherDataProvider, WeatherDataProvider
from model_engine.inputs.input_providers import DFTensorWeatherDataContainer
from model_engine.models.base_model import BatchTensorModel
from model_engine.models.states_rates import VariableKiosk


class BaseEngine(HasTraits):
    """
    Base Wrapper class for models
    """

    inputdataprovider = Instance(WeatherDataProvider, allow_none=True)

    def __init__(
        self,
        config: DictConfig = None,
        inputprovider: WeatherDataProvider = None,
        device: torch.device = "cpu",
    ) -> None:
        self.device = device
        self.config = config
        self.start_date = np.datetime64(config.start_date)
        self.day = self.start_date
        self.kiosk = VariableKiosk()

        self.output_vars = self.config["output_vars"]
        self.input_vars = self.config["input_vars"]

        self.model_constr = get_models(f"{os.path.dirname(os.path.abspath(__file__))}/models")[config.model]

        if inputprovider is None:
            self.inputdataprovider = NASAPowerWeatherDataProvider(self.config.latitude, self.config.longitude)
        else:
            self.inputdataprovider = inputprovider

    def run(self, dates: datetime.date = None, days: int = 1) -> torch.Tensor:
        """
        Advances the system state with given number of days
        """
        days_done = 0
        while days_done < days:
            days_done += 1
            self._run(date=dates)

        return self.get_output()

    def _run(self, date: datetime.date = None, cultivar: int = -1, delt: int = 1, **kwargs) -> None:
        """
        Helper function for run. Implemented by subclasses
        """
        raise NotImplementedError

    def run_all(self, end_date=datetime.date(2000, 9, 7), same_yr: bool = True) -> pd.DataFrame:
        """
        Run a simulation through termination
        Generally used to generate synthetic data
        """
        start_date = self.day.astype("datetime64[D]").astype(object)
        end_date = end_date.replace(year=start_date.year) if same_yr else end_date.replace(year=start_date.year + 1)
        df = pd.DataFrame(index=range((end_date - start_date).days), columns=self.output_vars + self.input_vars)

        inp = self.get_input(self.day)  # Do this first for correct ordering of input
        out = self.get_output().cpu().numpy().flatten()

        df.loc[0] = np.concatenate((out, inp))
        i = 1
        while self.day < end_date:
            inp = self.get_input(np.datetime64(self.day.astype("datetime64[D]").tolist() + datetime.timedelta(days=1)))
            out = self.run().cpu().numpy().flatten()
            df.loc[i] = np.concatenate((out, inp))
            i += 1
        return df

    def get_state_rates_names(self) -> list[str]:
        """
        Get names of valid states and ratse
        """
        return self.model.get_state_rates_names()

    def get_input(self, day: datetime.date) -> np.ndarray:
        """
        Get the input to a model on the day
        """

        return np.array([getattr(self.inputdataprovider(day), v) for v in self.input_vars], dtype=object)

    def get_output(self) -> torch.Tensor:
        """
        Get the output of a model
        """
        raise NotImplementedError

    def update_internal_model(self, new_state: torch.Tensor, changed_states: torch.Tensor) -> None:
        """
        Update internal model state
        """
        raise NotImplementedError

    def add_variables(self, drv: DFTensorWeatherDataContainer, **kwargs) -> DFTensorWeatherDataContainer:
        """
        Add additional variables to the daily driving variables to be passed to model
        """
        if "TRESP" in kwargs:
            if kwargs["TRESP"] is not None:
                temp_response = kwargs["TRESP"].flatten()
                temp_response = (
                    F.pad(
                        temp_response,
                        (0, self.num_models - len(temp_response)),
                        mode="constant",
                        value=0,
                    )
                    if len(temp_response) < self.num_models
                    else temp_response
                )
                drv.TRESP = temp_response

        if "ADDL" in kwargs:
            if kwargs["ADDL"] is not None:
                addv_latent_state = kwargs["ADDL"].flatten()
                addv_latent_state = (
                    F.pad(
                        addv_latent_state,
                        (0, self.num_models - len(addv_latent_state)),
                        mode="constant",
                        value=0,
                    )
                    if len(addv_latent_state) < self.num_models
                    else addv_latent_state
                )
                drv.ADDL = addv_latent_state

        return drv


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
        inputprovider: WeatherDataProvider = None,
        device: torch.device = "cpu",
    ) -> None:

        super().__init__(config, inputprovider, device)
        self.num_models = num_models
        self.model = self.model_constr(
            self.start_date,
            self.kiosk,
            param_loader(self.config),
            self.device,
            num_models=self.num_models,
        )
        assert isinstance(
            self.model, BatchTensorModel
        ), "Model specified is not a Batch Tensor Model, but we are using the BatchModelEngine as a wrapper!"

    def reset(self, num_models: int = 1, year: int = None, day: datetime.date = None) -> torch.Tensor:
        """
        Reset the model
        """
        if day is None:
            if year is not None:
                day = self.start_date.astype("M8[s]").astype(datetime.datetime).date()
                self.day = np.datetime64(day.replace(year=year))
            else:
                self.day = self.start_date
        else:
            self.day = day
        self.model.reset(self.day)

        return self.get_output()[:num_models]

    def run(self, dates: datetime.date = None, cultivars: list[int] = None, days: int = 1, **kwargs) -> torch.Tensor:
        """
        Advances the system state with given number of days
        """
        days_done = 0
        while days_done < days:
            days_done += 1
            self._run(dates=dates, cultivars=cultivars, **kwargs)
        return self.get_output()[: len(dates)] if dates is not None else self.get_output()

    def _run(
        self, dates: datetime.date = None, cultivars: torch.Tensor = None, delt: float = 1, **kwargs
    ) -> torch.Tensor:
        """
        Make one time step of the simulation.
        """
        if dates is None:
            self.day += np.timedelta64(1, "D")
            days = self.day
            drv = self.inputdataprovider(self.day, -1)
            drv.to_tensor(self.device)
        else:
            self.day = dates
            # Need to pad outputs to align with batch, we will ignore these in output
            if cultivars is None:
                days = (
                    np.pad(
                        self.day,
                        (0, self.num_models - len(self.day)),
                        mode="constant",
                        constant_values=self.day[-1],
                    )
                    if len(self.day) < self.num_models
                    else self.day
                )
                drv = self.inputdataprovider(days, np.tile(-1, len(days)))
            else:
                days = (
                    np.pad(
                        self.day,
                        (0, self.num_models - len(self.day)),
                        mode="constant",
                        constant_values=self.day[-1],
                    )
                    if len(self.day) < self.num_models
                    else self.day
                )
                cultivars = (
                    F.pad(
                        cultivars,
                        (0, 0, 0, self.num_models - len(cultivars)),
                        mode="constant",
                        value=float(cultivars[-1].cpu().numpy().flatten()),
                    )
                    if len(cultivars) < self.num_models
                    else cultivars
                )
                drv = self.inputdataprovider(days, cultivars)

        drv = self.add_variables(drv, **kwargs)

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

    def update_internal_model(self, new_state: torch.Tensor, changed_states: torch.Tensor) -> None:
        """
        Update the internal model with new state
        """
        new_state = (
            F.pad(
                new_state,
                (0, self.num_models - len(new_state)),
                mode="constant",
                value=0,
            )
            if len(new_state) < self.num_models
            else new_state
        )
        changed_states = (
            F.pad(
                changed_states,
                (0, self.num_models - len(changed_states)),
                mode="constant",
                value=False,
            )
            if len(changed_states) < self.num_models
            else changed_states
        )
        self.model.update_model(new_state, changed_states)

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


