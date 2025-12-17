"""
process_data.py

Contains functions to process phenology and cold hardiness data
for model training

Written by Will Solow, 2025
"""

import datetime
import numpy as np
import torch
from torch import nn

from model_engine.util import EPS, CULTIVARS



def process_data_inference(model: nn.Module) -> None:
    """Process all of the initial data"""

    model.output_vars = model.config.PConfig.output_vars
    model.input_vars = model.config.PConfig.input_vars

    model.params = model.config.params
    model.params_range = torch.tensor(np.array(model.config.params_range, dtype=np.float32)).to(model.device)

    model.num_cultivars = len(CULTIVARS)


def date_to_cyclic(date_str: str | datetime.date | np.ndarray) -> np.ndarray:
    """
    Convert datetime to cyclic embedding
    """
    if isinstance(date_str, np.ndarray):
        dt_arr = []
        for dt64 in date_str:
            dt = dt64.astype("datetime64[us]").item()
            day_of_year = dt.timetuple().tm_yday
            year_sin = np.sin(2 * np.pi * day_of_year / 365)
            year_cos = np.cos(2 * np.pi * day_of_year / 365)
            dt_arr.append([year_sin, year_cos])
        return np.array(dt_arr)
    else:
        if isinstance(date_str, str):
            date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        elif isinstance(date_str, datetime.date):
            date_obj = date_str
        else:
            msg = "Invalid type to convert to date"
            raise Exception(msg)

        day_of_year = date_obj.timetuple().tm_yday
        year_sin = np.sin(2 * np.pi * day_of_year / 365)
        year_cos = np.cos(2 * np.pi * day_of_year / 365)

        return np.array([year_sin, year_cos])


def normalize(data: torch.Tensor | np.ndarray, drange: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """
    Normalize data given a range
    """
    return (data - drange[:, 0]) / (drange[:, 1] + EPS)


def unnormalize(data: torch.Tensor | np.ndarray, drange: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """
    Unnormalize data given a range
    """
    return data * (drange[:, 1] + EPS) + drange[:, 0]
