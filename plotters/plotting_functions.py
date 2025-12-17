"""
functions.py

Contains tensorboard plotting functions and functions to plot
cold hardiness and phenology data

Written by Will Solow, 2025
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from omegaconf import DictConfig

from model_engine.util import PHENOLOGY_INT


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


def compute_RMSE_STAGE(true_output: np.ndarray, model_output: np.ndarray, stage: int) -> np.ndarray:

    curr_stage = (stage) % len(PHENOLOGY_INT)
    true_stage_args = np.argwhere(true_output == curr_stage).flatten()
    model_stage_args = np.argwhere(model_output == curr_stage).flatten()

    if len(true_stage_args) == 0 or len(model_stage_args) == 0:
        return (len(true_stage_args) + len(model_stage_args)) ** 2
    else:
        return (true_stage_args[0] - model_stage_args[0]) ** 2


def compute_rmse_plot(
    config: DictConfig,
    true: list[np.ndarray],
    model: list[np.ndarray],
    path: str,
    n_stages: int = 5,
    name: str = "Train",
    save: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Plot the errors in predicting onset of a phenological stage
    """
    avgs = np.zeros((n_stages, len(true))) - 1

    if isinstance(true, np.ndarray):
        true = [true]
        model = [model]

    for s in range(n_stages):
        for i in range(len(true)):
            true[i] = np.floor(true[i])
            model[i] = np.floor(model[i])
            if s not in true[i] and s not in model[i]:
                continue
            avgs[s, i] = compute_RMSE_STAGE(true[i], model[i], s)
    if avgs.shape[1] != 0:
        avg = np.sqrt(np.mean(np.ma.masked_equal(avgs, -1), axis=1)).filled(0)
        std = np.sqrt(np.std(np.ma.masked_equal(avgs, -1), axis=1)).filled(0)
    else:
        avg = np.zeros((n_stages))
        std = np.zeros((n_stages))

    if save:
        x = np.arange(n_stages)
        plt.figure()
        plt.bar(x, avg, label="Training Error")

        title = f"{config.DataConfig.cultivar} RMSE {name}"
        method = "RMSE"

        plt.title(title)
        plt.xlabel("Stage")
        plt.xticks(
            ticks=x,
            labels=["Ecodorm", "Budbreak", "Bloom", "Veraison", "Ripe"],
            rotation=0,
        )
        plt.ylabel("Average Error in Days")
        plt.savefig(f"{path}/PHENOLOGY_{method}_{name}_{config.DataConfig.cultivar}.png")
        plt.close()

    return avg, std


def plot_loss(fpath: str, config: DictConfig, tag1: str = "train_loss", tag2: str = "eval_loss") -> None:
    """
    Plot the training and testing losses
    """
    event_acc = EventAccumulator(fpath)
    event_acc.Reload()

    tag1_events = event_acc.Scalars(tag1)
    tag2_events = event_acc.Scalars(tag2)
    tag1_steps = [event.step for event in tag1_events]
    tag1_values = [event.value for event in tag1_events]
    tag2_steps = [event.step for event in tag2_events]
    tag2_values = [event.value for event in tag2_events]

    plt.figure()
    plt.plot(tag1_steps, tag1_values, label=tag1)
    plt.plot(tag2_steps, tag2_values, label=tag2)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{fpath}/Loss_{config.DataConfig.cultivar}.png", bbox_inches="tight")


def plot_stats(
    fpath: str,
    config: DictConfig,
    tag1: str = "model_grad_norm",
    tag2: str = "learning_rate",
    tag3: str = "weight_norm",
) -> None:
    """
    Plot the model_grad norm, learning rate, and weight norm
    throughout training
    """
    event_acc = EventAccumulator(fpath)
    event_acc.Reload()

    if tag1 is not None:
        tag1_events = event_acc.Scalars(tag1)
        tag1_steps = [event.step for event in tag1_events]
        tag1_values = [event.value for event in tag1_events]
        plt.figure()
        plt.plot(tag1_steps, tag1_values, label=tag1)
        plt.xlabel("Epochs")
        plt.ylabel("Gradient Norm")
        plt.legend()
        plt.savefig(f"{fpath}/GradNorm_{config.DataConfig.cultivar}.png", bbox_inches="tight")

    if tag2 is not None:
        tag2_events = event_acc.Scalars(tag2)
        tag2_steps = [event.step for event in tag2_events]
        tag2_values = [event.value for event in tag2_events]
        plt.figure()
        plt.plot(tag2_steps, tag2_values, label=tag2)
        plt.xlabel("Epochs")
        plt.ylabel("Learning Rate")
        plt.legend()
        plt.savefig(f"{fpath}/LearningRate_{config.DataConfig.cultivar}.png", bbox_inches="tight")

    if tag3 is not None:
        tag3_events = event_acc.Scalars(tag3)
        tag3_steps = [event.step for event in tag3_events]
        tag3_values = [event.value for event in tag3_events]
        plt.figure()
        plt.plot(tag3_steps, tag3_values, label=tag3)
        plt.xlabel("Epochs")
        plt.ylabel("Weight Norm of Parameters")
        plt.legend()
        plt.savefig(f"{fpath}/WeightNorm_{config.DataConfig.cultivar}.png", bbox_inches="tight")


def plot_params(fpath: str, config: DictConfig, params: np.ndarray, k: int, name: str = "Train") -> None:
    """
    Plot the predicted parameters for a time series
    """
    fig, ax = plt.subplots(params.shape[1], figsize=(6, 8))
    for i, p in enumerate(config.params):
        if params.shape[1] == 1:
            ax.plot(params[:, i], label=f"{p} Predicted")
            ax.set_ylabel(f"{p}")
            ax.set_xlim(0, params.shape[0])
            ax.set_title(f"Parameters {name} {k} {config.DataConfig.cultivar}")
            ax.set_xlabel("Days")
        else:
            ax[i].plot(params[:, i], label=f"{p} Predicted")
            ax[i].set_ylabel(f"{p}")
            ax[i].set_xlim(0, params.shape[0])
            ax[0].set_title(f"Parameters {name} {k} {config.DataConfig.cultivar}")
            ax[-1].set_xlabel("Days")
    plt.tight_layout()
    plt.savefig(f"{fpath}/Parameters_{name}_{k}_{config.DataConfig.cultivar}.png", bbox_inches="tight")
    plt.close()


def plot_output_phenology(
    config: DictConfig,
    fpath: str,
    i: np.ndarray,
    output: torch.Tensor,
    params: torch.Tensor,
    val_data: torch.Tensor,
    weather: np.ndarray = None,
    name: str = "Train",
    save: bool = True,
) -> torch.Tensor:
    """
    Plot the phenenology and the parameters for each time series in the passed batch
    """
    inds = ~torch.isnan(val_data.cpu().squeeze())
    val_data = val_data.cpu().numpy()

    if output.shape[-1] == len(PHENOLOGY_INT):  # Handle categorical classification
        output = torch.tensor(output)
        probs = F.softmax(output, dim=-1)
        output = torch.argmax(probs, dim=-1)

    # Handles case when batch size of 1 is passed
    if len(inds.shape) != 2:
        inds = inds[np.newaxis, :]
        if len(output.shape) != 2:
            output = output[np.newaxis, :]
            params = params[np.newaxis, :] if params is not None else None

    assert len(inds.shape) == 2, "Incorrectly specified data, ensure that the batch setting is being passed"
    if save:
        for k in range(inds.shape[0]):
            x = np.arange(len(output[k][inds[k]]))
            fig, ax = plt.subplots(2)
            target = config.PConfig.output_vars
            ax[0].plot(output[k][inds[k]], label=f"Model {target}")
            ax[0].plot(val_data[k][inds[k]], label=f"True {target}")
            ax[0].legend()
            ax[0].set_title(f"{target} (Predicted)")
            ax[0].set_xticks([], [])
            ax[0].set_xlim([0, len(x)])
            ax[0].set_ylabel(f"{target}")
            ax[0].set_yticks(
                [0, 1, 2, 3, 4],
                ["Ecodormancy", "Budbreak", "Bloom", "Veraison", "Ripe"],
            )
            ax[0].set_ylim([0, 5])

            # Plot predicted TBASEM vs True
            if params is not None:
                ax[1].plot(np.array(params[k])[:, 0][inds[k]], label="TBASEM Predicted")
                if (
                    config.DataConfig.cultivar is None
                    or config.DataConfig.cultivar == "synth"
                    or config.DataConfig.cultivar == "None"
                ):
                    ax[1].plot(x, np.tile(8.19, len(x)), label="True TBASEM")
                ax[1].set_title("TBASEM Parameter Prediction")
                ax[1].legend()
                ax[1].set_xlim([0, len(x)])
                ax[1].set_ylabel("TBASEM Value")
                ax[1].set_xlabel("Days")

            plt.savefig(
                f"{fpath}/Model_{name}_{i[k]}_{config.DataConfig.cultivar}.png",
                bbox_inches="tight",
            )
            plt.close()
            if params is not None:
                plot_params(fpath, config, np.array(params[k]), i[k], name=name)
    return inds


def plot_output_coldhardiness(
    config: DictConfig,
    fpath: str,
    i: np.ndarray,
    output: torch.Tensor,
    params: torch.Tensor,
    val_data: torch.Tensor,
    name: str = "Train",
    save: bool = True,
) -> torch.Tensor:
    """
    Plot the cold haridness and the parameters for each time series in the passed batch
    """
    ind_val_data = val_data.cpu().squeeze()
    inds = ~torch.isnan(ind_val_data[:, :, 0]) if ind_val_data.shape[-1] == 3 else ~torch.isnan(ind_val_data)
    val_data = val_data.cpu().numpy()
    val_data = val_data[:, :, 0] if val_data.shape[-1] == 3 else val_data  # Handle GCHN prediction
    output = output[:, :, 0] if output.shape[-1] == 3 else output

    # Handle when a batch size of 1 is passed
    if len(inds.shape) != 2:
        inds = inds[np.newaxis, :]
        if len(output.shape) != 2:
            output = output[np.newaxis, :]
            params = params[np.newaxis, :] if params is not None else None

    assert len(inds.shape) == 2, "Incorrectly specified data, ensure that the batch setting is being passed"
    if save:
        for k in range(inds.shape[0]):
            x = np.arange(len(output[k][inds[k]]))
            fig, ax = plt.subplots(2)
            target = config.PConfig.output_vars
            prop_cycle = plt.rcParams["axes.prop_cycle"]
            colors = prop_cycle.by_key()["color"]
            ax[0].plot(output[k], label=f"Model {target}", color=colors[0])
            ax[0].scatter(
                np.where(inds[k]),
                val_data[k][inds[k]],
                label=f"True {target}",
                s=10,
                marker="x",
                color=colors[1],
            )
            ax[0].legend()
            ax[0].set_title(f"{target} (Predicted)")
            ax[0].set_xticks([], [])
            ax[0].set_xlim([0, len(inds[k])])
            ax[0].set_ylabel(f"{target}")

            if params is not None:
                ax[1].plot(np.array(params[k])[:, 0], label="TENDO Predicted")
                ax[1].set_title("TENDO Parameter Prediction")
                ax[1].legend()
                ax[1].set_xlim([0, len(inds[k])])
                ax[1].set_ylabel("TENDO Value")
                ax[1].set_xlabel("Days")

            plt.savefig(
                f"{fpath}/Model_{name}_{i[k]}_{config.DataConfig.cultivar}.png",
                bbox_inches="tight",
            )
            plt.close()
            if params is not None:
                plot_params(fpath, config, np.array(params[k]), i[k], name=name)

    return inds
