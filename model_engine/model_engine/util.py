"""
util.py

Utility functions for the model_egnine class

Modified by Will Solow, 2024
"""

import yaml
import os
import torch
from inspect import getmembers, isclass
import importlib.util
import numpy as np

from model_engine.models.base_model import Model

EPS = 1e-12
PHENOLOGY_INT = {"Ecodorm": 0, "Budbreak": 1, "Bloom": 2, "Veraison": 3, "Ripe": 4}

# Available cultivars for simulation
CULTIVARS = {
    "grape_phenology_": np.array(
        [
            "Aligote",
            "Alvarinho",
            "Auxerrois",
            "Barbera",
            "Cabernet_Franc",
            "Cabernet_Sauvignon",
            "Chardonnay",
            "Chenin_Blanc",
            "Concord",
            "Durif",
            "Gewurztraminer",
            "Green_Veltliner",
            "Grenache",
            "Lemberger",
            "Malbec",
            "Melon",
            "Merlot",
            "Mourvedre",
            "Muscat_Blanc",
            "Nebbiolo",
            "Petit_Verdot",
            "Pinot_Blanc",
            "Pinot_Gris",
            "Pinot_Noir",
            "Riesling",
            "Sangiovese",
            "Sauvignon_Blanc",
            "Semillon",
            "Tempranillo",
            "Viognier",
            "Zinfandel",
        ],
        dtype=str,
    ),
    "grape_coldhardiness_": np.array(
        [
            "Barbera",
            "Cabernet_Franc",
            "Cabernet_Sauvignon",
            "Chardonnay",
            "Chenin_Blanc",
            "Concord",
            "Gewurztraminer",
            "Grenache",
            "Lemberger",
            "Malbec",
            "Merlot",
            "Mourvedre",
            "Nebbiolo",
            "Pinot_Gris",
            "Riesling",
            "Sangiovese",
            "Sauvignon_Blanc",
            "Semillon",
            "Syrah",
            "Viognier",
            "Zinfandel",
        ],
        dtype=str,
    ),
}


def param_loader(config: dict) -> dict:
    """
    Load the configuration of a model from dictionary
    """
    try:
        model_name, model_num = config["model_parameters"].split(":")
    except:
        raise Exception(f"Incorrectly specified model_parameters file `{config['model_parameters']}`")

    fname = f"{os.getcwd()}/{config['config_fpath']}{model_name}.yaml"
    try:
        model = yaml.safe_load(open(fname))
    except:
        raise Exception(f"Unable to load file: {fname}. Check that file exists")

    try:
        cv = model["ModelParameters"]["Sets"][model_num]
    except:
        raise Exception(
            f"Incorrectly specified parameter file {fname}. Ensure that `{model_name}` contains parameter set `{model_num}`"
        )

    for c in cv.keys():
        cv[c] = cv[c][0]

    return cv


def per_task_param_loader(config: dict, params: list) -> torch.Tensor:
    """
    Load the available configurations of a model from dictionary and put them on tensor
    """

    dtype = config.DataConfig.dtype.rsplit("_", 1)[0]
    fname = f"{os.getcwd()}/{config.PConfig.config_fpath}{dtype}.yaml"
    try:
        model = yaml.safe_load(open(fname))
    except:
        raise Exception(f"Unable to load file: {fname}. Check that file exists")
    init_params = []
    for n in CULTIVARS[config.DataConfig.dtype]:
        try:
            cv = model["ModelParameters"]["Sets"][n]
        except:
            raise Exception(
                f"Incorrectly specified parameter file {fname}. Ensure that `{config.DataConfig.dtype}` contains parameter set `{n}`"
            )
        task_params = []
        for c in params:
            if c in cv.keys():
                task_params.append(cv[c][0])
        init_params.append(task_params)

    return torch.tensor(init_params)


def get_models(folder_path: str) -> list[type]:
    """
    Get all the models in the /models/ folder
    """
    constructors = {}

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".py"):
            file_path = os.path.join(folder_path, filename)

            # Remove the .py extension
            module_name = filename[:-3]

            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            for name, obj in getmembers(module):
                if isclass(obj) and (issubclass(obj, Model)):
                    constructors[f"{name}"] = obj
        elif os.path.isdir(f"{folder_path}/{filename}"):
            constr = get_models(f"{folder_path}/{filename}")
            constructors = constructors | constr

    return constructors
