"""
utils.py

Declaration of arguments dataclasses and helper
functions for loading configurations

Written by Will Solow, 2025
"""

import os
import pickle
from typing import Optional
from dataclasses import dataclass
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
from torch import nn
from omegaconf import OmegaConf, DictConfig
from model_engine.util import CULTIVARS

from train_algs import DMC


@dataclass
class PConfig:
    """Path to configuration files"""

    config_fpath: Optional[str] = None
    """Model to use"""
    model: Optional[str] = None
    """Model parameters to use"""
    model_parameters: Optional[str] = None
    """Latitude of location"""
    latitude: Optional[int] = None
    """Longitude of location"""
    longitude: Optional[int] = None
    """Start date"""
    start_date: Optional[str] = None
    """Output Variables"""
    output_vars: Optional[list] = None
    """Input variables"""
    input_vars: Optional[list] = None


@dataclass
class DConfig:
    """Model Type"""

    type: Optional[str] = None
    """Model architecture"""
    arch: Optional[str] = None
    """Learning rate for training"""
    learning_rate: Optional[float] = None
    """Learning rate reduction factor"""
    lr_factor: Optional[float] = None
    """Batch size"""
    batch_size: Optional[int] = None
    """Epochs for training"""
    epochs: Optional[int] = None
    """Hidden dim of Model"""
    hidden_dim: Optional[int] = None


@dataclass
class DataConfig:
    """Path to real data"""

    data_fpath: Optional[str] = True
    """Data type (cold-hardiness, phenology, wheat)"""
    dtype: Optional[str] = None
    """Cultivar"""
    cultivar: Optional[str] = None


@dataclass
class Args:
    """Model configuration"""

    PConfig: object = PConfig
    """Deep Model Args"""
    DConfig: object = DConfig
    """Data Args"""
    DataConfig: object = DataConfig

    """Path to save model"""
    log_path: Optional[str] = None
    """Run name for experiment"""
    run_name: Optional[str] = None
    """Seed"""
    seed: Optional[int] = None
    """Parameters to predict"""
    params: Optional[list] = None
    """Ranges for each parameter to predict"""
    params_range: Optional[list] = None
    """Parameters smoothing coefficient"""
    param_scale: Optional[float] = None


def load_config_data(args: Namespace) -> tuple[DictConfig, list[pd.DataFrame]]:
    """
    Load the configuration and data and assumes
    that the configuration file is in _train_configs
    """

    config = OmegaConf.load(f"_train_configs/{args.config}.yaml")
    config = OmegaConf.merge(Args, config)
    config.seed = int(args.seed)

    if hasattr(args, "cultivar"):
        config.DataConfig.cultivar = args.cultivar if args.cultivar is not None else config.DataConfig.cultivar

    return config, load_data_from_config(config)


def load_config_data_fpath(
    args: Namespace,
) -> tuple[DictConfig, list[pd.DataFrame], str]:
    """
    Loads configuration file and data and returns
    filepath to load model
    """

    fpath = f"{os.getcwd()}/{args.config}"

    config = OmegaConf.load(f"{fpath}/config.yaml")
    config = OmegaConf.merge(Args, config)

    return config, load_data_from_config(config), fpath


def load_data_from_config(config: DictConfig) -> list[pd.DataFrame]:
    """
    Loads data from a OmegaConf configuration file
    """

    def find_pickle_files(root_dir: str, prefix: str = "", contains: str = "", exclude: str = None):
        """
        Recursively find all pickle files in root_dir and subdirectories
        that match a given prefix and contain a specified substring.

        Args:
            root_dir (str): The directory to search in.
            prefix (str): The filename prefix to match (default: "").
            contains (str): A substring that must appear in the filename (default: "").

        Returns:
            List[str]: List of full paths to matching pickle files.
        """
        pickle_paths = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                if exclude is not None:
                    if (
                        filename.endswith(".pkl")
                        and filename.startswith(prefix)
                        and contains in filename
                        and exclude not in filename
                    ):
                        full_path = os.path.join(dirpath, filename)
                        pickle_paths.append(full_path)
                else:
                    if filename.endswith(".pkl") and filename.startswith(prefix) and contains in filename:
                        full_path = os.path.join(dirpath, filename)
                        pickle_paths.append(full_path)
        return pickle_paths

    dtype = config.DataConfig.dtype.rsplit("_", 1)[0]

    PREFIX = f"{config.DataConfig.data_fpath}{dtype}"

    paths = find_pickle_files(
        f"{PREFIX}/WA/Roza2/Prosser/",
        contains="" if config.DataConfig.cultivar == "All" else config.DataConfig.cultivar,
        exclude="synth",
    )
    data = []

    for p in paths:
        cultivar = p.split(f"{dtype}_")[-1].replace(".pkl", "")
        if cultivar not in CULTIVARS[config.DataConfig.dtype]:
            continue
        with open(p, "rb") as f:
            cult_data = pickle.load(f)
        for cult in cult_data:
            cult["CULTIVAR"] = np.where(CULTIVARS[config.DataConfig.dtype] == cultivar)[0][0]
            data.append(cult)
    for d in data:
        d.rename(columns={"DATE": "DAY"}, inplace=True)

    return data


def load_model_from_config(config: DictConfig, data: list[pd.DataFrame]) -> nn.Module:
    """
    Load the model to train using the configuration and pass
    it args and data
    """

    if config.DConfig.type == "Param":
        calibrator = DMC.ParamRNN(config, data)
    elif config.DConfig.type == "NoObsParam":
        calibrator = DMC.NoObsParamRNN(config, data)
    else:
        raise NotImplementedError(f"Unrecognized RNN model type `{config.DConfig.type}`")

    return calibrator


def load_dfs(path: str) -> list[pd.DataFrame]:
    """
    Load dataframes from a filepath
    """
    df = pd.read_csv(path)

    df["DAY"] = pd.to_datetime(df["DAY"])

    inds = np.argwhere((df.DAY.dt.month == 1) & (df.DAY.dt.day == 1)).flatten()
    inds = np.concatenate((inds, [-1]))
    df_list = []
    for i in range(1, len(inds)):
        if inds[i] == -1:
            df_list.append(df.loc[inds[i - 1] :])
        else:
            df_list.append(df.loc[inds[i - 1] : inds[i] - 1])
    return df_list


def find_config_yaml_dirs(start_dir="."):
    config_dirs = []
    for root, dirs, files in os.walk(start_dir):
        if "config.yaml" in files:
            relative_path = os.path.relpath(root, start_dir)
            config_dirs.append(relative_path)
    return config_dirs


def load_named_pickles(folder_paths: list[str], target_name: str, include: str = None, exclude_multi: bool = False):
    """
    Load all pickle files matching a given name in all subdirectories.
    """
    results = {}

    for root in folder_paths:
        root = Path(f"./_runs/{root}")

        for pkl_file in root.rglob(target_name):
            try:
                # Get relative subdirectory name
                subdir = "/".join(pkl_file.parent.parts[-6:])
                if include is not None and not (include in subdir):
                    continue
                if "All" in subdir and exclude_multi:
                    continue
                with open(pkl_file, "rb") as f:
                    results[subdir] = pickle.load(f)
            except Exception as e:
                pass

    return results
