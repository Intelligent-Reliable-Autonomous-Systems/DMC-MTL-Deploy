"""
util.py

Contains utility functions for the input provider and nasapower classes

Written by Will Solow, 2025
Specific functions came from https://github.com/ajwdewit/pcse
"""

import datetime
from math import radians
import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################################################
# Used for Crop models to calculate daylength for development
###############################################################


def doy(day: np.datetime64 | datetime.date | datetime.datetime) -> int:
    """
    Converts a date or datetime object to day-of-year (Jan 1st = doy 1)
    """
    # Check if day is a date or datetime object
    if isinstance(day, (datetime.date, datetime.datetime)):
        return day.timetuple().tm_yday
    elif isinstance(day, np.datetime64):
        return day.astype("datetime64[D]").tolist().timetuple().tm_yday
    else:
        msg = "Parameter day is not a date or datetime object."
        raise RuntimeError(msg)


def daylength(day: list | np.ndarray, latitude: float | torch.Tensor, angle: float = -4) -> np.ndarray:
    """
    Calculates the daylength for a given day, altitude and base.

    :param day:         date/datetime object
    :param latitude:    latitude of location
    :param angle:       The photoperiodic daylength starts/ends when the sun
        is `angle` degrees under the horizon. Default is -4 degrees.

    Derived from the WOFOST routine ASTRO.FOR and simplified to include only
    daylength calculation. Results are being cached for performance.
    Modified to handle tensors and numpy arrays
    """
    if isinstance(latitude, torch.Tensor):
        if latitude.ndim == 0:
            latitude = latitude.unsqueeze(0)

    if isinstance(latitude, float):
        if abs(latitude) > 90.0:
            msg = "Latitude not between -90 and 90"
            raise RuntimeError(msg)
    else:
        # Check for range of latitude
        if (abs(latitude) > 90.0).any():
            msg = "Latitude not between -90 and 90"
            raise RuntimeError(msg)

    # Calculate day-of-year from date object day
    if isinstance(day, list) or isinstance(day, np.ndarray):
        IDAY = np.array([doy(d) for d in day])
    else:
        IDAY = doy(day)

    # constants
    RAD = radians(1.0)

    # calculate daylength
    ANGLE = angle
    if isinstance(latitude, float):
        LAT = latitude
    elif isinstance(latitude, np.ndarray):
        LAT = latitude.flatten()
    else:
        LAT = latitude.cpu().squeeze()
    DEC = -np.arcsin(np.sin(23.45 * RAD) * np.cos(2.0 * np.pi * (IDAY + 10.0) / 365.0))
    SINLD = np.sin(RAD * LAT) * np.sin(DEC)
    COSLD = np.cos(RAD * LAT) * np.cos(DEC)
    AOB = (-np.sin(ANGLE * RAD) + SINLD) / COSLD

    # daylength
    DAYLP = np.where(
        np.abs(AOB) <= 1.0,
        12.0 * (1.0 + 2.0 * np.arcsin((-np.sin(ANGLE * RAD) + SINLD) / COSLD) / np.pi),
        np.where(AOB > 1.0, 24.0, 0.0),
    )
    return DAYLP
