"""
plot_utils.py

Contains data generation functions used in plotting interfaces

Written by Will Solow, 2025
"""

from argparse import Namespace
from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

from model_engine.util import PHENOLOGY_INT
from plotters.plotting_functions import (
    plot_output_phenology,
    plot_output_coldhardiness,
    plot_output_wofost,
    plot_output_coldhardiness_error,
)


def compute_obs_RMSE(true: list[np.ndarray], model: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes RMSE of per observation
    """
    if isinstance(true, np.ndarray):
        true = [true]
        model = [model]

    max_len = 0
    for t in true:
        max_len = len(t) if len(t) > max_len else max_len

    samples = np.zeros(shape=max_len)
    avgs = np.zeros(shape=max_len)

    for i in range(len(true)):
        for j in range(len(true[i])):
            avgs[j] += (true[i][j] - model[i][j]) ** 2
            samples[j] += 1

    samples = np.where(samples == 0, 1, samples)  # ignore divde by zero

    return np.sqrt(avgs / samples), None


def compute_total_RMSE(true: list[np.ndarray], model: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes total RMSE across entire dataset
    """

    avgs = 0
    samples = 0
    if isinstance(true, np.ndarray):
        true = [true]
        model = [model]

    for i in range(len(true)):
        avgs += np.sum((true[i] - model[i]) ** 2)
        samples += len(true[i])
    if samples == 0:
        return 0, None
    else:
        return np.sqrt(avgs / samples), None


def compute_day_RMSE(
    true: list[np.ndarray], model: list[np.ndarray], inds: list[np.ndarray], days: int = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes RMSE up to a given day
    """

    avgs = 0
    samples = 0
    if isinstance(true, np.ndarray):
        true = [true]
        model = [model]

    for i in range(len(true)):
        k = (inds[i] >= days) * (inds[i] < (days + 30))
        avgs += np.sum((true[i][k] - model[i][k]) ** 2)
        samples += np.sum(k)
    if samples == 0:
        return 0
    else:

        return np.sqrt(avgs / samples)


def gen_batch_data(
    calibrator: nn.Module,
    input_data: torch.Tensor,
    dates: np.ndarray,
    val_data: torch.Tensor,
    cultivars: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates model output from a single batch
    """
    with torch.no_grad():
        (output, params, _) = calibrator.forward(
            input_data,
            dates,
            cultivars=cultivars,
            val_data=val_data,
        )

    true_data = val_data.cpu().squeeze().numpy()
    if output.size(-1) == len(PHENOLOGY_INT):
        probs = F.softmax(output, dim=-1)
        output = torch.argmax(probs, dim=-1)
    output_data = output.detach().cpu().squeeze().numpy()

    params = params.cpu().numpy().squeeze() if params is not None else None

    return true_data, output_data, params


def gen_all_data_and_plot(
    config: DictConfig,
    fpath: str,
    args: Namespace,
    calibrator: nn.Module,
    true_data: list[np.ndarray],
    output_data: list[np.ndarray],
    true_cultivar_data: list[np.ndarray],
    output_cultivar_data: list[np.ndarray],
    name: str = "train",
    days: int = None,
    all_inds: list[np.ndarray] = None,
    cult_inds: list[np.ndarray] = None,
) -> None:
    """
    Generates data for train and testing data and plots accordingly for RNNs
    """
    if name == "train":
        n = 0
    elif name == "val":
        n = 1
    else:  # test
        n = 2

    true_arr = []
    output_arr = []
    for i in range(0, len(calibrator.data[name]), calibrator.batch_size):

        cultivars = (
            calibrator.cultivars[name][i : i + calibrator.batch_size] if calibrator.cultivars is not None else None
        )
        true, output, params = gen_batch_data(
            calibrator,
            calibrator.data[name][i : i + calibrator.batch_size],
            calibrator.dates[name][i : i + calibrator.batch_size],
            calibrator.val[name][i : i + calibrator.batch_size],
            cultivars,
        )
        true_arr.append(true)
        output_arr.append(output)

        if "grape_phenology" in config.DataConfig.dtype:
            inds = plot_output_phenology(
                config,
                fpath,
                np.arange(start=i, stop=i + calibrator.batch_size),
                output,
                params,
                calibrator.val[name][i : i + calibrator.batch_size],
                name=name,
                save=args.save,
            )
        elif "grape_coldhardiness" in config.DataConfig.dtype:
            inds = plot_output_coldhardiness(
                config,
                fpath,
                np.arange(start=i, stop=i + calibrator.batch_size),
                output,
                params,
                calibrator.val[name][i : i + calibrator.batch_size],
                name=name,
                save=args.save,
            )
        elif "wofost" in config.DataConfig.dtype:
            inds = plot_output_wofost(
                config,
                fpath,
                np.arange(start=i, stop=i + calibrator.batch_size),
                output,
                params,
                calibrator.val[name][i : i + calibrator.batch_size],
                name=name,
                save=args.save,
            )

        if len(true.shape) == 1:
            true = true[np.newaxis, :]
            output = output[np.newaxis, :]
        if true.shape[-1] == 3:
            true = true[:, :, 0]
            output = output[:, :, 0]

        [true_data[n].append(true[k][inds[k]]) for k in range(len(true))]
        [output_data[n].append(output[k][inds[k]]) for k in range(len(output))]
        (
            [all_inds[n].append(torch.argwhere(inds[k]).flatten().cpu().numpy()) for k in range(len(inds))]
            if all_inds is not None
            else None
        )

        cm = calibrator.nn.cult_mapping if hasattr(calibrator.nn, "cult_mapping") else [0, 0]

        for k in range(len(true)):
            ck = int(cultivars[k].item()) if cultivars is not None else 0

            true_cultivar_data[cm[ck]][n].append(true[k][inds[k]])
            output_cultivar_data[cm[ck]][n].append(output[k][inds[k]])
            (
                cult_inds[cm[ck]][n].append(torch.argwhere(inds[k]).flatten().cpu().numpy())
                if cult_inds is not None
                else None
            )

        if args.break_early:
            break
