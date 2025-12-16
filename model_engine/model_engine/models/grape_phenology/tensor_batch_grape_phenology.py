"""
tensor_batch_grape_phenology.py

Implementation of the grape phenology model based on the GDD model
with pytorch tensors to simulate multiple models on batch

Written by Will Solow, 2025
"""

import datetime
import torch
import numpy as np

from model_engine.inputs.weather_util import daylength
from model_engine.models.base_model import BatchTensorModel
from model_engine.models.states_rates import Tensor, NDArray
from model_engine.models.states_rates import (
    ParamTemplate,
    StatesTemplate,
    RatesTemplate,
    VariableKiosk,
)
from model_engine.inputs.input_providers import DFTensorWeatherDataContainer

EPS = 1e-12


class Grape_Phenology_TensorBatch(BatchTensorModel):
    """Implements grape phenology GDD model"""

    # Based on the Elkhorn-Lorenz Grape Phenology Stage
    _STAGE_VAL = {
        "ecodorm": 0,
        "budbreak": 1,
        "bloom": 2,
        "veraison": 3,
        "ripe": 4,
        "endodorm": 5,
    }

    _STAGE = NDArray(["ecodorm"])
    _DAY_LENGTH = Tensor(12.0)  # Helper variable for daylength

    class Parameters(ParamTemplate):
        TBASEM = Tensor(-99.0)  # Base temp. for bud break
        TEFFMX = Tensor(-99.0)  # Max eff temperature for grow daily units
        TSUMEM = Tensor(-99.0)  # Temp. sum for bud break

        TSUM1 = Tensor(-99.0)  # Temperature sum budbreak to bloom
        TSUM2 = Tensor(-99.0)  # Temperature sum bloom to veraison
        TSUM3 = Tensor(-99.0)  # Temperature sum from veraison to ripe
        TSUM4 = Tensor(-99.0)  # Temperature sum from ripe onwards
        MLDORM = Tensor(-99.0)  # Daylength at which a plant will go into dormancy
        Q10C = Tensor(-99.0)  # Parameter for chilling unit accumulation
        CSUMDB = Tensor(-99.0)  # Chilling unit sum for dormancy break

    class RateVariables(RatesTemplate):
        DTSUME = Tensor(-99.0)  # increase in temperature sum for emergence
        DTSUM = Tensor(-99.0)  # increase in temperature sum
        DVR = Tensor(-99.0)  # development rate
        DCU = Tensor(-99.0)  # Daily chilling units

    class StateVariables(StatesTemplate):
        PHENOLOGY = Tensor(-0.99)  # Int of Stage
        DVS = Tensor(-99.0)  # Development stage
        TSUME = Tensor(-99.0)  # Temperature sum for emergence state
        TSUM = Tensor(-99.0)  # Temperature sum state
        CSUM = Tensor(-99.0)  # Chilling sum state

    def __init__(
        self,
        day: datetime.date,
        kiosk: VariableKiosk,
        parvalues: dict,
        device: torch.device,
        num_models: int = 1,
    ) -> None:
        self.num_models = num_models
        self.num_stages = len(self._STAGE_VAL)
        self.stages = list(self._STAGE_VAL.keys())
        super().__init__(day, kiosk, parvalues, device, num_models=self.num_models)

        # Define initial states
        self._STAGvarsE = ["ecodorm" for _ in range(self.num_models)]
        self.states = self.StateVariables(
            num_models=self.num_models,
            TSUM=0.0,
            TSUME=0.0,
            DVS=0.0,
            CSUM=0.0,
            PHENOLOGY=self._STAGE_VAL["ecodorm"],
            device=self.device,
        )

        self.rates = self.RateVariables(num_models=self.num_models, device=self.device)
        self.min_tensor = torch.tensor([0.0]).to(self.device)

    def calc_rates(self, day: datetime.date, drv: DFTensorWeatherDataContainer) -> None:
        """
        Calculates the rates for phenological development
        """
        p = self.params
        r = self.rates
        # Day length sensitivity
        if hasattr(drv, "DAYL"):
            self._DAY_LENGTH = drv.DAYL
        elif hasattr(drv, "LAT"):
            self._DAY_LENGTH = daylength(day, drv.LAT)
        if self._DAY_LENGTH.ndim == 0:
            self._DAY_LENGTH = torch.tile(self._DAY_LENGTH, (self.num_models,))[: self.num_models]
        elif len(self._DAY_LENGTH) < self.num_models:
            self._DAY_LENGTH = torch.tile(self._DAY_LENGTH, (self.num_models // len(self._DAY_LENGTH) + 1,))[
                : self.num_models
            ]

        r.DTSUME = torch.zeros(size=(self.num_models,)).to(self.device)
        r.DTSUM = torch.zeros(size=(self.num_models,)).to(self.device)
        r.DVR = torch.zeros(size=(self.num_models,)).to(self.device)

        stage_tensor = torch.tensor([self._STAGE_VAL[s] for s in self._STAGE], device=self.device)
        stage_masks = torch.stack([stage_tensor == i for i in range(self.num_stages)])
        (
            self._ecodorm,
            self._budbreak,
            self._bloom,
            self._veraison,
            self._ripe,
            self._endodorm,
        ) = stage_masks

        # Compute DTSUM values. If the DRV has a temperature response function
        # then use that instead
        if hasattr(drv, "TRESP"):
            dtsum_update = torch.clamp(drv.TRESP, self.min_tensor, p.TEFFMX)
        else:
            dtsum_update = torch.clamp(drv.TEMP - p.TBASEM, self.min_tensor, p.TEFFMX)
        # Apply DTSUM updates only where each stage condition is met
        r.DTSUM = torch.where(
            self._endodorm | self._budbreak | self._bloom | self._veraison | self._ripe,
            dtsum_update,
            r.DTSUM,
        )
        r.DTSUME = torch.where(self._ecodorm, dtsum_update, r.DTSUME)

        # Stack all TSUM values for vectorized selection
        TSUM_stack = torch.where(
            self._ripe,
            p.TSUM4,
            torch.where(
                self._veraison,
                p.TSUM3,
                torch.where(
                    self._bloom,
                    p.TSUM2,
                    torch.where(
                        self._budbreak,
                        p.TSUM1,
                        torch.where(
                            self._ecodorm,
                            p.TSUMEM,
                            torch.where(self._endodorm, p.TSUM4, torch.ones_like(p.TSUM4)),
                        ),
                    ),
                ),
            ),
        )

        r.DVR = torch.where(self._ecodorm, r.DTSUME / TSUM_stack.clamp(min=EPS), r.DTSUM / TSUM_stack.clamp(min=EPS))

        # Add latent internal state if present in arguments
        if hasattr(drv, "ADDL"):
            r.DVR = r.DVR + drv.ADDL

    def integrate(self, day, delt=1.0):
        """
        Updates the state variable and checks for phenologic stages
        """

        p = self.params
        r = self.rates
        s = self.states

        # Integrate phenologic states
        s.TSUME = s.TSUME + r.DTSUME
        s.DVS = s.DVS + r.DVR

        s.TSUM = s.TSUM + r.DTSUM
        s.CSUM = s.CSUM + r.DCU
        s.PHENOLOGY = torch.floor(s.DVS).detach() + (s.DVS - s.DVS.detach())

        # Stage transitions
        self._STAGE[(self._endodorm & (s.CSUM >= p.CSUMDB)).cpu().numpy()] = "ecodorm"
        s.TSUM[self._endodorm & (s.CSUM >= p.CSUMDB)] = 0.0
        s.TSUME[self._endodorm & (s.CSUM >= p.CSUMDB)] = 0.0
        s.DVS[self._endodorm & (s.CSUM >= p.CSUMDB)] = 0.0
        s.CSUM[self._endodorm & (s.CSUM >= p.CSUMDB)] = 0.0

        self._STAGE[(self._ecodorm & (s.TSUME >= p.TSUMEM)).cpu().numpy()] = "budbreak"
        self._STAGE[(self._budbreak & (s.DVS >= 2.0)).cpu().numpy()] = "bloom"
        self._STAGE[(self._bloom & (s.DVS >= 3.0)).cpu().numpy()] = "veraison"
        self._STAGE[(self._veraison & (s.DVS >= 4.0)).cpu().numpy()] = "ripe"
        self._STAGE[(self._veraison & (self._DAY_LENGTH <= p.MLDORM)).cpu().numpy()] = "endodorm"
        self._STAGE[(self._ripe & (self._DAY_LENGTH <= p.MLDORM)).cpu().numpy()] = "endodorm"

    def get_output(self, va: list = None) -> torch.Tensor:
        """
        Return the phenological stage as the floor value
        """
        if va is None:
            return torch.unsqueeze(self.states.DVS, -1)
        else:
            output_vars = torch.empty(size=(self.num_models, len(va))).to(self.device)
            for i, v in enumerate(va):
                if v in self.states.trait_names():
                    output_vars[:, i] = getattr(self.states, v)
                elif v in self.rates.trait_names():
                    output_vars[:, i] = getattr(self.rates, v)
            return output_vars

    def reset(self, day: datetime.date, inds: torch.Tensor = None) -> None:
        """
        Reset the model
        """
        # Define initial states
        s = self.states
        r = self.rates
        inds = torch.ones((self.num_models,), device=self.device).to(torch.bool) if inds is None else inds

        self._STAGE = np.where(inds.cpu().numpy(), "ecodorm", self._STAGE)

        s.TSUM = torch.where(inds, 0.0, s.TSUM).detach()
        s.TSUME = torch.where(inds, 0.0, s.TSUME).detach()
        s.DVS = torch.where(inds, 0.0, s.DVS).detach()
        s.CSUM = torch.where(inds, 0.0, s.CSUM).detach()
        s.PHENOLOGY = torch.where(inds, self._STAGE_VAL["ecodorm"], s.PHENOLOGY).detach()

        r.DTSUME = torch.where(inds, 0.0, r.DTSUME).detach()
        r.DTSUM = torch.where(inds, 0.0, r.DTSUM).detach()
        r.DVR = torch.where(inds, 0.0, r.DVR).detach()
        r.DCU = torch.where(inds, 0.0, r.DCU).detach()

        s._update_kiosk()
        r._update_kiosk()

    def update_model(self, new_state: torch.Tensor, changed_states: torch.Tensor) -> None:
        """
        Update the model with new passed state
        """
        s = self.states
        s.DVS = torch.where(changed_states, new_state, s.DVS)
        s.PHENOLOGY = torch.floor(s.DVS).detach() + (s.DVS - s.DVS.detach())

    def get_extra_states(self) -> dict[str, torch.Tensor]:
        """
        Get extra states
        """
        return {"_STAGE": self._STAGE, "_DAY_LENGTH": self._DAY_LENGTH}

    def set_model_specific_params(self, k: str, v: torch.Tensor) -> None:
        """
        Set the specific parameters to handle overrides as needed
        Like casting to ints
        """
        setattr(self.params, k, v)
