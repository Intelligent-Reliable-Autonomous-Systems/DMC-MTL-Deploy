"""
process_data.py

Contains functions to process phenology and cold hardiness data
for model training

Written by Will Solow, 2025
"""

import datetime
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from omegaconf import DictConfig

from model_engine.util import EPS, CULTIVARS
from model_engine.inputs.input_providers import (
    MultiTensorWeatherDataProvider,
    WeatherDataProvider,
)


def process_data(model: nn.Module, data: list[pd.DataFrame]) -> None:
    """Process all of the initial data"""

    if len(data) < 3:
        raise Exception(f"Data size: {len(data)}. Insufficient data for building training set.")

    model.output_vars = model.config.PConfig.output_vars
    model.input_vars = model.config.PConfig.input_vars

    model.params = model.config.params
    model.params_range = torch.tensor(np.array(model.config.params_range, dtype=np.float32)).to(model.device)

    # Get normalized (weather) data
    normalized_input_data, model.drange = embed_and_normalize_zscore([d.loc[:, model.input_vars] for d in data])

    normalized_input_data = pad_sequence(normalized_input_data, batch_first=True, padding_value=0).to(model.device)
    model.drange = model.drange.to(torch.float32).to(model.device)

    # Get input data for use with model to avoid unnormalizing
    extra_feats = ["CULTIVAR"]
    extra_feats = extra_feats + ["DAY"] if "DAY" not in model.input_vars else extra_feats
    model.input_data = make_tensor_inputs(model.config, [d.loc[:, model.input_vars + extra_feats] for d in data])

    # Get validation data
    output_data, output_range = embed_output([d.loc[:, model.output_vars] for d in data])
    output_data = pad_sequence(output_data, batch_first=True, padding_value=model.target_mask).to(model.device)
    model.output_range = output_range.to(torch.float32).to(model.device)

    # Get the dates
    dates = [d.loc[:, "DAY"].to_numpy().astype("datetime64[D]") for d in data]
    max_len = max(len(arr) for arr in dates)
    # Pad each array to the maximum length
    dates = [np.pad(arr, (0, max_len - len(arr)), mode="maximum") for arr in dates]
    x = 2  # Number of years for testing set

    # Shuffle to get train and test splits for data
    train_inds = np.empty(shape=(0,))
    test_inds = np.empty(shape=(0,))
    cultivar_data = np.array([d.loc[0, "CULTIVAR"] for d in data]) if "CULTIVAR" in data[0].columns else None

    for c in range(len(CULTIVARS[model.config.DataConfig.dtype])):
        cultivar_inds = np.argwhere((c == cultivar_data)).flatten()
        if len(cultivar_inds) < 3:
            continue

        np.random.shuffle(cultivar_inds)
        test_inds = np.concatenate((test_inds, cultivar_inds[:x])).astype(np.int32)

        train_inds = np.concatenate((train_inds, cultivar_inds[x:][:])).astype(np.int32)

    np.random.shuffle(train_inds)
    np.random.shuffle(test_inds)
    model.data = {
        "train": torch.stack([normalized_input_data[i] for i in train_inds]).to(torch.float32),
        "test": (
            torch.stack([normalized_input_data[i] for i in test_inds]).to(torch.float32)
            if len(test_inds) > 0
            else torch.tensor([])
        ),
    }
    model.val = {
        "train": torch.stack([output_data[i] for i in train_inds]).to(torch.float32),
        "test": (
            torch.stack([output_data[i] for i in test_inds]).to(torch.float32)
            if len(test_inds) > 0
            else torch.tensor([])
        ),
    }

    model.dates = {
        "train": np.array([dates[i] for i in train_inds]),
        "test": (np.array([dates[i] for i in test_inds]) if len(test_inds) > 0 else np.array([])),
    }

    cultivar_data = (
        torch.tensor(cultivar_data).to(torch.float32).to(model.device).unsqueeze(1)
        if cultivar_data is not None
        else None
    )

    model.num_cultivars = len(torch.unique(cultivar_data)) if cultivar_data is not None else 1

    model.cultivars = {
        "train": torch.stack([cultivar_data[i] for i in train_inds]).to(torch.float32),
        "test": torch.stack([cultivar_data[i] for i in test_inds]).to(torch.float32),
    }

    if len(model.data["test"]) < 1:
        raise Exception("Insuffient per-cultivar data to build test set")


def make_tensor_inputs(config: DictConfig, dfs: list[pd.DataFrame]) -> WeatherDataProvider:
    """
    Make input providers based on the given data frames
    Converts data frames to tensor table
    """

    wp = MultiTensorWeatherDataProvider(pd.concat(dfs, ignore_index=True))

    return wp


def embed_and_normalize_zscore(
    data: list[pd.DataFrame],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Embed and normalize all data using z-score normalization
    Handle if "DAY" is present in the first entry
    """
    tens = []
    stacked_data = (
        np.vstack([d.to_numpy()[:, 1:] for d in data]).astype(np.float32)
        if "DAY" in data[0].columns
        else np.vstack([d.to_numpy() for d in data]).astype(np.float32)
    )
    data_mean = np.nanmean(stacked_data, axis=0).astype(np.float32)
    data_std = np.std(stacked_data, axis=0).astype(np.float32)

    if "DAY" in data[0].columns:
        data_mean = np.concatenate(([0, 0], data_mean)).astype(np.float32)
        data_std = np.concatenate(([1 / np.sqrt(2), 1 / np.sqrt(2)], data_std)).astype(np.float32)

    for d in data:
        d = d.to_numpy()
        if "DAY" in data[0].columns:
            dt = np.reshape([date_to_cyclic(d[i, 0]) for i in range(len(d[:, 0]))], (-1, 2))
            d = np.concatenate((dt, d[:, 1:]), axis=1).astype(np.float32)
        # Z-score normalization
        d = (d - data_mean) / (data_std + EPS)
        tens.append(torch.tensor(d.astype(np.float32), dtype=torch.float32))

    return tens, torch.tensor(np.stack((data_mean, data_std), axis=-1))


def embed_output(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns output data and mean, std to normalize if needed
    """
    tens = []
    stacked_data = np.vstack([d.to_numpy() for d in data]).astype(np.float32)
    data_mean = np.nanmean(stacked_data, axis=0).astype(np.float32)
    data_std = np.nanstd(stacked_data, axis=0).astype(np.float32)  # This used to be std

    for d in data:
        d = d.to_numpy()
        tens.append(torch.tensor(d, dtype=torch.float32))

    return tens, torch.tensor(np.stack((data_mean, data_std), axis=-1))


def date_to_cyclic(date_str: str | datetime.date) -> list[np.ndarray]:
    """
    Convert datetime to cyclic embedding
    """
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

    return [year_sin, year_cos]
