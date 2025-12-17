"""
input_providers.py

Handles making weather providers for models.

Written by Will Solow, 2025
Original inspiration from https://github.com/ajwdewit/pcse
but has since been heavily modified
"""

import torch
import numpy as np
import datetime as dt
import pickle
import pandas as pd


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


class WeatherDataProvider(object):
    """
    Base class for all weather data providers.
    """

    longitude = None
    latitude = None
    elevation = None
    description = []
    _first_date = None
    _last_date = None
    angstA = None
    angstB = None
    ETmodel = "PM"

    def __init__(self, device: str = "cpu") -> None:
        self.store = {}
        self.device = device

    def _dump(self, cache_fname: str) -> None:
        """Dumps the contents into cache_fname using pickle.

        Dumps the values of self.store, longitude, latitude, elevation and description
        """
        with open(cache_fname, "wb") as fp:
            dmp = (
                self.store,
                self.elevation,
                self.longitude,
                self.latitude,
                self.description,
                self.ETmodel,
            )
            pickle.dump(dmp, fp, pickle.HIGHEST_PROTOCOL)

    def _load(self, cache_fname: str) -> None:
        """Loads the contents from cache_fname using pickle.

        Loads the values of self.store, longitude, latitude, elevation and description
        from cache_fname and also sets the self.first_date, self.last_date
        """
        with open(cache_fname, "rb") as fp:
            (
                store,
                self.elevation,
                self.longitude,
                self.latitude,
                self.description,
                ETModel,
            ) = pickle.load(fp)

        # Check if the reference ET from the cache file is calculated with the same model as
        # specified by self.ETmodel
        if ETModel != self.ETmodel:
            msg = "Mismatch in reference ET from cache file."
            raise Exception(msg)

        self.store.update(store)

    @property
    def first_date(self) -> dt.date:
        try:
            self._first_date = min(self.store)[0]
        except ValueError:
            pass
        return self._first_date

    @property
    def last_date(self) -> dt.date:
        try:
            self._last_date = max(self.store)[0]
        except ValueError:
            pass
        return self._last_date

    def check_keydate(
        self, key: dt.datetime | dt.date | str | int | np.datetime64 | np.ndarray | list
    ) -> dt.datetime | np.ndarray:
        """Check representations of date for storage/retrieval of weather data.

        The following formats are supported:

        1. a date object
        2. a datetime object
        3. a string of the format YYYYMMDD
        4. a string of the format YYYYDDD

        Formats 2-4 are all converted into a date object internally.
        """

        import datetime as dt

        if isinstance(key, dt.datetime):
            return key.date()
        elif isinstance(key, dt.date):
            return key
        elif isinstance(key, (str, int)):
            date_formats = {7: "%Y%j", 8: "%Y%m%d", 10: "%Y-%m-%d"}
            skey = str(key).strip()
            l = len(skey)
            if l not in date_formats:
                msg = "Key for WeatherDataProvider not recognized as date: %s"
                raise KeyError(msg % key)

            dkey = dt.datetime.strptime(skey, date_formats[l])
            return dkey.date()
        elif isinstance(key, np.datetime64):
            return key.astype("datetime64[D]").tolist()
        elif isinstance(key, np.ndarray):
            return np.array([self.check_keydate(k) for k in key])
        elif isinstance(key, list):
            return np.array([self.check_keydate(k) for k in key])
        else:
            msg = "Key for WeatherDataProvider not recognized as date: %s"
            raise KeyError(msg % key)

    def _store_WeatherDataContainer(
        self,
        wdc: SlotPickleMixin,
        keydate: dt.datetime | dt.date | str | int,
        cultivar: int = -1,
    ) -> None:
        """Stores the WDC under given keydate."""
        kd = self.check_keydate(keydate)
        self.store[kd, cultivar] = wdc

    def __call__(self, day: dt.datetime | dt.date | str | int, cultivar: int = -1) -> DFTensorWeatherDataContainer:
        keydate = self.check_keydate(day)
        try:
            if isinstance(keydate, np.ndarray):
                slots = self.store[keydate[0], cultivar[0]].__slots__
                if len(slots) == 0:
                    self.store[keydate[0], cultivar[0]].__slots__ = list(
                        self.store[keydate[0], cultivar[0]].__dict__.keys()
                    )
                    slots = self.store[keydate[0], cultivar[0]].__slots__
                vals = dict(
                    zip(
                        slots,
                        [np.empty(shape=len(keydate), dtype=object)]
                        + [torch.empty(size=(len(keydate),)).to(self.device) for _ in range(len(slots) - 1)],
                    )
                )
                for i, key in enumerate(keydate):
                    weather = self.store[key, cultivar[i]]
                    for s in slots:
                        vals[s][i] = getattr(weather, s)
                return DFTensorWeatherDataContainer(device=self.device, **vals)
            else:
                return self.store[keydate, cultivar]
        except KeyError as e:
            msg = "No weather data for %s." % keydate
            raise Exception(msg)


class MultiTensorWeatherDataProvider(WeatherDataProvider):
    """
    Creates lookup table of weather tensors and
    then each day aggregates and returns a DFTensorWeatherDataContainer
    for the daily weather in the batch setting
    """

    def __init__(self, df: pd.DataFrame = None, device: str = "cpu") -> None:

        WeatherDataProvider.__init__(self, device=device)

        if df is not None:
            self._get_and_process_DF(df)

    def _get_and_process_DF(self, df: pd.DataFrame) -> None:
        """
        Handles the retrieval and processing of the NASA Power data
        """

        if "CULTIVAR" in df.columns:
            self.keys = dict(
                zip(
                    zip(
                        np.datetime_as_string(df["DAY"].to_numpy().astype("datetime64[D]"), "D").tolist(),
                        df["CULTIVAR"].to_numpy().astype(int).tolist(),
                    ),
                    range(len(df["DAY"])),
                )
            )
            self.attrs = df.drop(columns=["DAY", "CULTIVAR"], inplace=False).columns.to_list()
            self.values = (
                torch.tensor(df.drop(columns=["DAY", "CULTIVAR"], inplace=False).to_numpy())
                .to(torch.float32)
                .to(self.device)
            )
        else:
            self.keys = dict(
                zip(
                    zip(
                        np.datetime_as_string(df["DAY"].to_numpy().astype("datetime64[D]"), "D").tolist(),
                        [-1] * len(df["DAY"]),
                    ),
                    range(len(df["DAY"])),
                )
            )
            self.attrs = df.drop(columns=["DAY"], inplace=False).columns.to_list()
            self.values = torch.tensor(df.drop(columns=["DAY"], inplace=False).to_numpy()).to(torch.float32).to(DEVICE)

    def _dump(self, cache_fname: str) -> None:
        """Dumps the contents into cache_fname using pickle.

        Dumps the values of self.store, longitude, latitude, elevation and description
        """
        with open(cache_fname, "wb") as fp:
            dmp = (self.keys, self.values.cpu().numpy(), self.attrs)
            pickle.dump(dmp, fp, pickle.HIGHEST_PROTOCOL)

    def _load(self, cache_fname: str) -> None:
        """Loads the contents from cache_fname using pickle.

        Loads the values of self.store, longitude, latitude, elevation and description
        from cache_fname and also sets the self.first_date, self.last_date
        """
        with open(cache_fname, "rb") as fp:
            (self.keys, self.values, self.attrs) = pickle.load(fp)
            self.values = torch.tensor(self.values).to(self.device)

    def check_keydate(self, key: dt.datetime | dt.date | np.datetime64 | np.ndarray | list) -> str:
        """Check representations of date for storage/retrieval of weather data.

        The following formats are supported:

        1. a date object
        2. a datetime object
        3. a string of the format YYYYMMDD
        4. a string of the format YYYYDDD

        Formats 2-4 are all converted into a date object internally.
        """
        import datetime as dt

        if isinstance(key, dt.datetime):
            return key.date().strftime("%Y-%m-%d")
        elif isinstance(key, dt.date):
            return key.strftime("%Y-%m-%d")
        elif isinstance(key, np.datetime64):
            return key.astype("datetime64[D]").astype(str)
        elif isinstance(key, np.ndarray):
            return key.astype(str)
        elif isinstance(key, list):
            return np.array(key).astype(str)
        else:
            msg = "Key for WeatherDataProvider not recognized as date: %s"
            raise KeyError(msg % key)

    def check_cultivar(self, cultivar: int | float | str | np.ndarray | torch.Tensor) -> int | list:
        """Check for cultivar"""
        if isinstance(cultivar, int):
            return cultivar
        elif isinstance(cultivar, float):
            return int(cultivar)
        elif isinstance(cultivar, str):
            return int(cultivar)
        elif isinstance(cultivar, np.ndarray):
            return np.squeeze(cultivar).astype(int).tolist()
        elif isinstance(cultivar, torch.Tensor):
            return cultivar.squeeze().cpu().numpy().astype(int).tolist()
        else:
            msg = "Key for WeatherDataProvider not recognized as cultivar: %s"
            raise KeyError(msg % cultivar)

    def __call__(
        self,
        day: dt.datetime | dt.date | np.datetime64 | np.ndarray | list,
        cultivar: int = -1,
    ) -> DFTensorWeatherDataContainer:
        keydate = self.check_keydate(day)
        cultivar = self.check_cultivar(cultivar)
        try:
            if isinstance(keydate, np.ndarray):
                if not isinstance(cultivar, list):
                    cultivar = [cultivar] * len(keydate)
                inds = [self.keys[keydate[i], cultivar[i]] for i in range(len(keydate))]
                vals = torch.split(self.values[inds], 1, dim=1)
                vals = dict(zip(self.attrs, [v.squeeze() for v in vals]))
                return DFTensorWeatherDataContainer(device=self.device, **vals)
            else:
                inds = self.keys[keydate, cultivar]
                vals = torch.split(self.values[inds], 1, dim=0)
                vals = dict(zip(self.attrs, [v.squeeze() for v in vals]))
                return DFTensorWeatherDataContainer(device=self.device, **vals)
        except KeyError as e:
            msg = "No weather data for %s." % keydate
            raise Exception(msg)
