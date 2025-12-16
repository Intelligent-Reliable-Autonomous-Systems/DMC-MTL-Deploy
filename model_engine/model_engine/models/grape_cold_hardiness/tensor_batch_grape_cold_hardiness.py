"""
tensor_batch_grape_coldhardiness.py
Implementation of Feguson Model for Grape Cold Hardiness supporting batches

Written by Will Solow, 2025
"""

import datetime
import torch
import numpy as np

from model_engine.models.base_model import BatchTensorModel
from model_engine.models.states_rates import Tensor, NDArray
from model_engine.models.states_rates import (
    ParamTemplate,
    StatesTemplate,
    RatesTemplate,
    VariableKiosk,
)
from model_engine.inputs.input_providers import DFTensorWeatherDataContainer


class Grape_ColdHardiness_TensorBatch(BatchTensorModel):

    _STAGE_VAL = {"endodorm": 0, "ecodorm": 1}
    _STAGE = NDArray(["endodorm"])
    _HC_YESTERDAY = Tensor(-99.0)

    class Parameters(ParamTemplate):
        HCINIT = Tensor(-99.0)  # Initial Cold Hardiness
        HCMIN = Tensor(-99.0)  # Minimum cold hardiness (negative)
        HCMAX = Tensor(-99.0)  # Maximum cold hardiness (negative)
        TENDO = Tensor(-99.0)  # Endodorm temp
        TECO = Tensor(-99.0)  # Ecodorm temp
        ENACCLIM = Tensor(-99.0)  # Endo rate of acclimation
        ECACCLIM = Tensor(-99.0)  # Eco rate of acclimation
        ENDEACCLIM = Tensor(-99.0)  # Endo rate of deacclimation
        ECDEACCLIM = Tensor(-99.0)  # Eco rate of deacclimation
        THETA = Tensor(-99.0)  # Theta param for acclimation
        ECOBOUND = Tensor(-99.0)  # Temperature threshold for onset of ecodormancy
        LTE10M = Tensor(-99.0)  # Regression coefficient for LTE10
        LTE10B = Tensor(-99.0)  # Regression coefficient for LTE10
        LTE90M = Tensor(-99.0)  # Regression coefficient for LTE90
        LTE90B = Tensor(-99.0)  # Regression coefficient for LTE90

    class RateVariables(RatesTemplate):
        DCU = Tensor(-99.0)  # Daily heat accumulation
        DHR = Tensor(-99.0)  # Daily heating rate
        DCR = Tensor(-99.0)  # Daily chilling rate
        DACC = Tensor(-99.0)  # Deacclimation rate
        ACC = Tensor(-99.0)  # Acclimation rate
        HCR = Tensor(-99.0)  # Change in acclimation

    class StateVariables(StatesTemplate):
        CSUM = Tensor(-99.0)  # Daily temperature sum for phenology
        DHSUM = Tensor(-99.0)  # Daily heating sum
        DCSUM = Tensor(-99.0)  # Daily chilling sum
        HC = Tensor(-99.0)  # Cold hardiness
        PREDBB = Tensor(-99.0)  # Predicted bud break
        LTE50 = Tensor(-99.0)  # Predicted LTE50 for cold hardiness
        LTE10 = Tensor(-99.0)  # Predicted LTE10 for cold hardiness
        LTE90 = Tensor(-99.0)  # Predicted LTE90 for cold hardiness

    def __init__(
        self,
        day: datetime.date,
        kiosk: VariableKiosk,
        parvalues: dict,
        device: torch.device,
        num_models: int = 1,
    ) -> None:

        super().__init__(day, kiosk, parvalues, device, num_models=num_models)

        # Define initial states
        p = self.params
        self.num_models = num_models
        self.num_stages = len(self._STAGE_VAL)
        self.stages = list(self._STAGE_VAL.keys())
        self._STAGE = ["endodorm" for _ in range(self.num_models)]
        LTE10 = p.HCINIT * p.LTE10M + p.LTE10B
        LTE90 = p.HCINIT * p.LTE90M + p.LTE90B

        self.states = self.StateVariables(
            num_models=self.num_models,
            DHSUM=0.0,
            DCSUM=0.0,
            HC=p.HCINIT[0].detach().cpu(),
            PREDBB=0.0,
            LTE50=p.HCINIT[0].detach().cpu(),
            CSUM=0.0,
            LTE10=LTE10[0].detach().cpu(),
            LTE90=LTE90[0].detach().cpu(),
            device=self.device,
        )

        self.rates = self.RateVariables(num_models=self.num_models, device=self.device)
        self.min_tensor = torch.tensor([0.0]).to(self.device)
        self.base_tensor = torch.tensor([10.0]).to(self.device)
        self._HC_YESTERDAY = p.HCINIT[0].detach().clone()

    def calc_rates(self, day: datetime.date, drv: DFTensorWeatherDataContainer) -> None:
        """Calculates the rates for phenological development"""
        p = self.params
        r = self.rates
        s = self.states

        r.DCU = torch.zeros(size=(self.num_models,))
        r.DHR = torch.zeros(size=(self.num_models,))
        r.DCR = torch.zeros(size=(self.num_models,))
        r.DACC = torch.zeros(size=(self.num_models,))
        r.ACC = torch.zeros(size=(self.num_models,))
        r.HCR = torch.zeros(size=(self.num_models,))
        if hasattr(drv, "TRESP"):
            r.DCU = torch.min(self.min_tensor, drv.TRESP)
        else:
            r.DCU = torch.min(self.min_tensor, drv.TEMP - self.base_tensor)

        stage_tensor = torch.tensor([self._STAGE_VAL[s] for s in self._STAGE], device=self.device)  # create masks
        stage_masks = torch.stack([stage_tensor == i for i in range(self.num_stages)])  # one hot encoding matrix
        self._endodorm, self._ecodorm = stage_masks  # upack for readability

        r.DHR = torch.where(
            self._endodorm,
            torch.max(self.min_tensor, drv.TEMP - p.TENDO),
            torch.where(
                self._ecodorm,
                torch.max(self.min_tensor, drv.TEMP - p.TECO),
                torch.ones_like(p.TECO),
            ),
        )
        r.DCR = torch.where(
            self._endodorm,
            torch.min(self.min_tensor, drv.TEMP - p.TENDO),
            torch.where(
                self._ecodorm,
                torch.min(self.min_tensor, drv.TEMP - p.TECO),
                torch.ones_like(p.TECO),
            ),
        )

        r.DACC = torch.where(
            self._endodorm,
            torch.where(
                s.DCSUM != 0,
                r.DHR * p.ENDEACCLIM * (1 - ((self._HC_YESTERDAY - p.HCMAX) / (p.HCMIN - p.HCMAX).clamp(min=1e-6))),
                0,
            ),
            torch.where(
                self._ecodorm,
                torch.where(
                    s.DCSUM != 0,
                    r.DHR
                    * p.ECDEACCLIM
                    * (1 - ((self._HC_YESTERDAY - p.HCMAX) / (p.HCMIN - p.HCMAX).clamp(min=1e-6)) ** p.THETA),
                    0,
                ),
                torch.ones_like(p.HCMAX),
            ),
        )

        r.ACC = torch.where(
            self._endodorm,
            r.DCR * p.ENACCLIM * (1 - ((p.HCMIN - self._HC_YESTERDAY)) / ((p.HCMIN - p.HCMAX).clamp(min=1e-6))),
            torch.where(
                self._ecodorm,
                r.DCR * p.ECACCLIM * (1 - ((p.HCMIN - self._HC_YESTERDAY)) / ((p.HCMIN - p.HCMAX).clamp(min=1e-6))),
                torch.ones_like(p.HCMAX),
            ),
        )

        r.HCR = r.DACC + r.ACC

    def integrate(self, day: datetime.date, delt: float = 1.0) -> None:
        """
        Updates the state variable and checks for phenologic stages
        """

        p = self.params
        r = self.rates
        s = self.states

        # Integrate phenologic states
        s.CSUM = s.CSUM + r.DCU
        s.HC = torch.clamp(p.HCMAX, p.HCMIN, s.HC + r.HCR)
        self._HC_YESTERDAY = s.HC
        s.DCSUM = s.DCSUM + r.DCR
        s.LTE50 = s.HC
        s.LTE10 = s.LTE50 * p.LTE10M + p.LTE10B
        s.LTE90 = s.LTE50 * p.LTE90M + p.LTE90B

        # Use HCMIN to determine if vinifera or labrusca
        s.PREDBB = torch.where(
            (s.HC >= -2.2) & (self._HC_YESTERDAY < -2.2) & (p.HCMIN == -1.2),
            s.HC,
            torch.where(
                (s.HC >= -6.4) & (self._HC_YESTERDAY < -6.4) & (p.HCMIN == -2.5),
                s.HC,
                torch.zeros_like(s.HC),
            ),
        )

        # Check if a new stage is reached
        self._STAGE[(self._endodorm & (s.CSUM <= p.ECOBOUND)).cpu().numpy()] = "ecodorm"

    def get_output(self, va: list[str] = None) -> torch.Tensor:
        """
        Return the LTE50
        """
        if va is None:
            return torch.unsqueeze(self.states.LTE50, -1)
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

        p = self.params
        s = self.states
        r = self.rates
        inds = torch.ones((self.num_models,), device=self.device).to(torch.bool) if inds is None else inds

        self._STAGE = np.where(inds.cpu().numpy(), "endodorm", self._STAGE)

        LTE10 = p.HCINIT * p.LTE10M + p.LTE10B
        LTE90 = p.HCINIT * p.LTE90M + p.LTE90B
        s.DHSUM = torch.where(inds, 0.0, s.DCSUM).detach()
        s.DCSUM = torch.where(inds, 0.0, s.DCSUM).detach()
        s.HC = torch.where(inds, p.HCINIT.detach(), s.HC).detach()
        s.PREDBB = torch.where(inds, 0.0, s.PREDBB).detach()
        s.LTE50 = torch.where(inds, p.HCINIT.detach(), s.LTE50).detach()
        s.CSUM = torch.where(inds, 0.0, s.CSUM).detach()
        s.LTE10 = torch.where(inds, LTE10.detach(), s.LTE10).detach()

        s.LTE90 = torch.where(inds, LTE90.detach(), s.LTE90).detach()

        self._HC_YESTERDAY = torch.where(inds, p.HCINIT.detach(), self._HC_YESTERDAY).detach()

        r.DCU = torch.where(inds, 0.0, r.DCU).detach()
        r.DHR = torch.where(inds, 0.0, r.DHR).detach()
        r.DCR = torch.where(inds, 0.0, r.DCR).detach()
        r.DACC = torch.where(inds, 0.0, r.DACC).detach()
        r.ACC = torch.where(inds, 0.0, r.ACC).detach()
        r.HCR = torch.where(inds, 0.0, r.HCR).detach()

        s._update_kiosk()
        r._update_kiosk()

    def get_extra_states(self) -> dict[str, torch.Tensor]:
        """Get extra states"""
        return {"_STAGE": self._STAGE, "_HC_YESTERDAY": self._HC_YESTERDAY}

    def set_model_specific_params(self, k: str, v: torch.Tensor) -> None:
        """Set the specific parameters to handle overrides as needed
        Like casting to ints
        """
        if k == "THETA":
            setattr(self.params, k, torch.floor(v).detach() + (v - v.detach()))
        else:
            setattr(self.params, k, v)
