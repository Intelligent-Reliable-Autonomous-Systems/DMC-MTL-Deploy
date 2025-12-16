"""Base class for for State Rates and Parameters that each simulation object
in the WOFOST model has.

In general these classes are not to be used directly, but are to be subclassed
when creating PCSE simulation units.

Written by: Allard de Wit (allard.dewit@wur.nl), April 2014
Modified by Will Solow, 2024
"""

from traitlets_pcse import Float, Int, Instance, Bool, HasTraits, TraitType, All
import torch
import numpy as np
from collections.abc import Iterable


class Tensor(TraitType):
    """An AFGEN table trait"""

    default_value = torch.tensor([0.0])
    into_text = "An AFGEN table of XY pairs"

    def validate(self, obj: object, value: torch.Tensor | Iterable | float | int) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(torch.float32)
        elif isinstance(value, Iterable):
            return torch.tensor(value, dtype=torch.float32)
        elif isinstance(value, float):
            return torch.tensor([value], dtype=torch.float32)
        elif isinstance(value, int):
            return torch.tensor([float(value)], dtype=torch.float32)
        self.error(obj, value)


class NDArray(TraitType):
    """An AFGEN table trait"""

    default_value = torch.tensor([0.0])
    into_text = "An AFGEN table of XY pairs"

    def validate(self, obj: object, value: np.ndarray | Iterable | float | int) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        elif isinstance(value, Iterable):
            return np.array(value, dtype=object)
        elif isinstance(value, float):
            return np.array([value])
        elif isinstance(value, int):
            return np.array([value])
        self.error(obj, value)


class VariableKiosk(dict):
    """VariableKiosk for registering and publishing state variables in PCSE."""

    def __init__(self) -> None:
        """Initialize the class `VariableKiosk`"""
        dict.__init__(self)
        self.published_states = []
        self.published_rates = []

    def __setitem__(self, item: str, value: torch.Tensor) -> None:
        msg = "See set_variable() for setting a variable."
        raise RuntimeError(msg)

    def __contains__(self, item: str) -> bool:
        """Checks if item is in self.registered_states or self.registered_rates."""
        return dict.__contains__(self, item)

    def __getattr__(self, item: str) -> torch.Tensor:
        """Allow use of attribute notation (eg "kiosk.LAI") on published rates or states."""
        if item.startswith("__") and item.endswith("__"):
            raise AttributeError(f"{item} not found")
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(f"{item} not found") from e

    def __str__(self) -> str:
        msg = "Contents of VariableKiosk:\n"
        msg += " * Published state variables: %i with values:\n" % len(self.published_states)
        for varname in self.published_states:
            if varname in self:
                value = self[varname]
            else:
                value = "undefined"
            msg += "  - variable %s, value: %s\n" % (varname, value)
        msg += " * Published rate variables: %i with values:\n" % len(self.published_rates)
        for varname in self.published_rates:
            if varname in self:
                value = self[varname]
            else:
                value = "undefined"
            msg += "  - variable %s, value: %s\n" % (varname, value)
        return msg

    def register_variable(self, varname: str, type: str) -> None:
        """Register a varname from object with id, with given type"""
        if type.upper() == "R":
            self.published_rates.append(varname)
        elif type.upper() == "S":
            self.published_states.append(varname)
        else:
            pass

    def set_variable(self, varname: str, value: torch.Tensor) -> None:
        """Let set the value of variable varname"""

        if varname in self.published_rates:
            dict.__setitem__(self, varname, value)
        elif varname in self.published_states:
            dict.__setitem__(self, varname, value)
        else:
            msg = "Variable '%s' not published in VariableKiosk."
            raise Exception(msg % varname)

    def variable_exists(self, varname: str) -> bool:
        """Returns True if the state/rate variable is registered in the kiosk."""

        if varname in self.published_rates or varname in self.published_states:
            return True
        else:
            return False


class ParamTemplate(HasTraits):
    """
    Template for storing parameter values.
    """

    def __init__(self, parvalues: dict, num_models: int = None, device: str = "cpu") -> None:
        """Initialize parameter template
        Args:
            parvalues - parameter values to include
        """
        HasTraits.__init__(self)
        self._device = device

        for parname in self.trait_names():
            # Check if the parname is available in the dictionary of parvalues
            if parname not in parvalues:
                msg = "Value for parameter %s missing." % parname
                raise Exception(msg)
            if num_models is None:
                value = parvalues[parname]
            else:
                value = np.tile(parvalues[parname], num_models).astype(np.float32)
                if isinstance(parvalues[parname], list):
                    value = np.reshape(value, (num_models, -1))
            # Single value parameter
            setattr(self, parname, torch.tensor(value).to(self._device))

    def __setattr__(self, attr: str, value: torch.Tensor) -> None:
        if attr.startswith("_"):
            HasTraits.__setattr__(self, attr, value)
        elif hasattr(self, attr):
            HasTraits.__setattr__(self, attr, value)
        else:
            msg = "Assignment to non-existing attribute '%s' prevented." % attr
            raise Exception(msg)

    def __str__(self) -> str:
        string = f""
        for parname in self.trait_names():
            string += f"{parname}: {getattr(self, parname)}\n"
        return string


class StatesRatesCommon(HasTraits):
    """
    Base class for States/Rates Templates. Includes all commonalitities
    between the two templates
    """

    _valid_vars = Instance(set)

    def __init__(self, kiosk: VariableKiosk = None, publish: list = [], device: str = "cpu") -> None:
        """Set up the common stuff for the states and rates template
        including variables that have to be published in the kiosk
        """

        HasTraits.__init__(self)

        # Determine the rate/state attributes defined by the user
        self._valid_vars = self._find_valid_variables()
        self.device = device
        self._kiosk = kiosk
        self._published_vars = []
        if self._kiosk is not None:
            self._register_with_kiosk(publish)

    def _find_valid_variables(self) -> set:
        """
        Returns a set with the valid state/rate variables names. Valid rate
        variables have names not starting with 'trait' or '_'.
        """

        valid = lambda s: not (s.startswith("_") or s.startswith("trait"))
        r = [name for name in self.trait_names() if valid(name)]
        return set(r)

    def __str__(self) -> str:
        string = f""
        for parname in self.trait_names():
            string += f"{parname}: {getattr(self, parname)}\n"
        return string

    def _register_with_kiosk(self, publish: list = []) -> None:
        """Register the variable with the variable kiosk."""
        for attr in self._valid_vars:
            if attr in publish:
                self._published_vars.append(attr)
                self._kiosk.register_variable(attr, type=self._vartype)

    def _update_kiosk(self) -> None:
        """Update kiosk based on published vars"""
        for attr in self._published_vars:
            self._kiosk.set_variable(attr, getattr(self, attr))


class StatesTemplate(StatesRatesCommon):
    """
    Takes care of assigning initial values to state variables
    and monitoring assignments to variables that are published.
    """

    _vartype = "S"

    def __init__(
        self,
        kiosk: VariableKiosk = None,
        publish: list = [],
        num_models: int = None,
        device: str = "cpu",
        **kwargs: dict,
    ) -> None:
        """Initialize the StatesTemplate class

        Args:
            kiosk - VariableKiosk to handle default parameters
        """
        StatesRatesCommon.__init__(self, kiosk=kiosk, publish=publish, device=device)

        # set initial state value
        for attr in self._valid_vars:
            if attr in kwargs:
                value = kwargs.pop(attr)
                if num_models is None:
                    setattr(self, attr, value)
                else:
                    if isinstance(value, torch.Tensor):
                        if value.ndim == 0:
                            setattr(self, attr, torch.tile(value, (num_models,)).to(torch.float32).to(self.device))
                        else:
                            if value.size(0) == num_models:
                                setattr(self, attr, value)
                            else:
                                setattr(self, attr, torch.tile(value, (num_models,)).to(torch.float32).to(self.device))
                    else:
                        setattr(
                            self, attr, torch.tile(torch.tensor(value), (num_models,)).to(torch.float32).to(self.device)
                        )
            else:
                msg = "Initial value for state %s missing." % attr
                raise Exception(msg)

        self._update_kiosk()


class RatesTemplate(StatesRatesCommon):
    """
    Takes care of registering variables in the kiosk and monitoring
    assignments to variables that are published.
    """

    _vartype = "R"

    def __init__(
        self, kiosk: VariableKiosk = None, publish=[], num_models: int = None, device: str = "cpu", **kwargs: dict
    ) -> None:
        """Set up the RatesTemplate and set monitoring on variables that
        have to be published.
        """
        self.num_models = num_models
        StatesRatesCommon.__init__(self, kiosk=kiosk, publish=publish, device=device)

        # Determine the zero value for all rate variable if possible
        self._rate_vars_zero = self._find_rate_zero_values()

        # Initialize all rate variables to zero or False
        self.zerofy()

        self._update_kiosk()

    def _find_rate_zero_values(self) -> dict:
        """Returns a dict with the names with the valid rate variables names as keys and
        the values are the zero values used by the zerofy() method. This means 0 for Int,
        0.0 for Float en False for Bool.
        """

        # Define the zero value for Float, Int and Bool
        if self.num_models is None:
            tensor = torch.tensor([0.0]).to(self.device)
        else:
            tensor = torch.tensor(np.tile(0.0, self.num_models).astype(np.float32)).to(self.device)
        zero_value = {Bool: False, Int: 0, Float: 0.0, Tensor: tensor}

        d = {}
        for name, value in self.traits().items():
            if name not in self._valid_vars:
                continue
            try:
                d[name] = zero_value[value.__class__]
            except KeyError:
                continue
        return d

    def zerofy(self) -> None:
        """
        Sets the values of all rate values to zero (Int, Float)
        or False (Boolean).
        """
        self._trait_values.update(self._rate_vars_zero)
