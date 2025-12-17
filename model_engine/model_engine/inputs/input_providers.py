"""
input_providers.py

Handles making weather providers for models.

Written by Will Solow, 2025
Original inspiration from https://github.com/ajwdewit/pcse
but has since been heavily modified
"""

import torch
import numpy as np

class SlotPickleMixin(object):
    """This mixin makes it possible to pickle/unpickle objects with __slots__ defined.

    In many programs, one or a few classes have a very large number of instances.
    Adding __slots__ to these classes can dramatically reduce the memory footprint
    and improve execution speed by eliminating the instance dictionary. Unfortunately,
    the resulting objects cannot be pickled. This mixin makes such classes pickleable
    again and even maintains compatibility with pickle files created before adding
    __slots__.

    Recipe taken from:
    http://code.activestate.com/recipes/578433-mixin-for-pickling-objects-with-__slots__/
    """

    def __getstate__(self) -> dict:
        return dict((slot, getattr(self, slot)) for slot in self.__slots__ if hasattr(self, slot))

    def __setstate__(self, state: dict) -> None:
        for slot, value in state.items():
            setattr(self, slot, value)


class DFTensorWeatherDataContainer(SlotPickleMixin):
    """
    Class for storing weather data elements.
    """

    __slots__ = []

    def __init__(self, device: str = "cpu", *args: list, **kwargs: dict) -> None:

        self.__slots__ = list(kwargs.keys())
        self.device = device

        # only keyword parameters should be used for weather data container
        if len(args) > 0:
            msg = (
                "WeatherDataContainer should be initialized by providing weather "
                + "variables through keywords only. Got '%s' instead."
            )
            raise Exception(msg % args)
        # Set all attributes
        for k, v in kwargs.items():
            if isinstance(v, float) or isinstance(v, int):
                setattr(self, k, torch.tensor([v]).to(self.device))
            elif isinstance(v, torch.Tensor):
                setattr(self, k, v.to(self.device))
            elif isinstance(v, np.ndarray):
                setattr(self, k, torch.tensor(v).to(self.device))
            else:
                setattr(self, k, v)

    def __setattr__(self, key: str, value: torch.Tensor) -> None:
        SlotPickleMixin.__setattr__(self, key, value)

