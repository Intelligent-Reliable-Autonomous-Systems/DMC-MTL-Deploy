"""
util.py

Utility Files for training algorithms and data processing

Written by Will Solow, 2025
"""

import time
import os
from argparse import Namespace
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf

from model_engine.util import PHENOLOGY_INT


def set_embedding_op(model: nn.Module) -> int:
    """
    Set the embedding operation to be used
    in MultiTask Embedding Models
    """

    def concat(embed, input):
        return torch.concatenate((embed, input), dim=-1)

    model.embed_op = concat
    return 2 * model.input_dim


def get_grad_norm(model: nn.Module) -> float:
    """
    Get gradients of model
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm**0.5


def compute_RMSE_STAGE_tensor(
    true_output: torch.Tensor,
    model_output: torch.Tensor,
    mask: torch.Tensor = None,
    stage: int = 0,
) -> torch.Tensor:
    """
    Compute the RMSE of a stage
    """
    curr_stage = (stage) % len(PHENOLOGY_INT)
    if mask is None:
        mask = torch.ones(len(true_output))

    true_stage_args = torch.argwhere((true_output == curr_stage) * mask).flatten()
    model_stage_args = torch.argwhere((model_output == curr_stage) * mask).flatten()

    if len(true_stage_args) == 0 or len(model_stage_args) == 0:
        return (len(true_stage_args) + len(model_stage_args)) ** 2
    else:
        return (true_stage_args[0] - model_stage_args[0]) ** 2


def cumulative_error(
    true: torch.Tensor,
    model: torch.Tensor,
    mask: torch.Tensor = None,
    n_stages: int = 5,
    RMSE: bool = True,
    device: torch.device = "cpu",
) -> torch.Tensor:
    """
    Plot the errors in predicting onset of stage
    """
    avgs = torch.zeros(size=(n_stages,)).to(device)
    if model.size(-1) == 3:
        return avgs

    if model.size(-1) == len(PHENOLOGY_INT):  # Handle categorical classification
        probs = F.softmax(model, dim=-1)
        model = torch.argmax(probs, dim=-1)
    model = torch.round(model)  # Round to nearest integer for comparison
    if mask is None:
        mask = torch.ones(shape=true.shape).to(device)
    for s in range(n_stages):
        for i in range(len(true)):
            if RMSE:
                avgs[s] += compute_RMSE_STAGE_tensor(
                    true[i].flatten(),
                    model[i].flatten(),
                    mask=mask[i].flatten(),
                    stage=s,
                )
    return avgs


def setup_logging(config: Namespace) -> tuple[SummaryWriter, str]:
    """Setup Tensorboard Logging and W&B"""

    run_name = f"{config.run_name}__{int(time.time())}"
    log_path = f"{os.getcwd()}{config.log_path}/{config.DataConfig.cultivar}/{run_name}"

    writer = SummaryWriter(log_path)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in OmegaConf.to_container(config).items()])),
    )
    return writer, run_name, log_path


def log_training(
    calibrator: nn.Module,
    writer: SummaryWriter,
    fpath: str,
    epoch: int,
    train_loss: float,
    eval_loss: float,
    train_avg: float,
    eval_avg: float,
    grad: float,
) -> None:
    """
    Log training statistics and print to console
    """

    # RMSE
    eval_len = len(calibrator.data["test"])
    train_avg[:3] = torch.sqrt(train_avg[:3] / len(calibrator.data["train"]))
    eval_avg[:3] = torch.sqrt(eval_avg[:3] / eval_len)
    train_avg[-1] = torch.sum(train_avg[:3])
    eval_avg[-1] = torch.sum(eval_avg[:3])

    if hasattr(calibrator, "nn"):
        weight_norm = torch.sqrt(sum((p.data**2).sum() for p in calibrator.nn.parameters()))
    elif hasattr(calibrator, "finetuner"):
        weight_norm = torch.sqrt(sum((p.data**2).sum() for p in calibrator.finetuner.parameters()))

    writer.add_scalar("train_loss", train_loss, epoch)
    writer.add_scalar("eval_loss", eval_loss, epoch)
    for k in range(4):
        writer.add_scalar(f"train_rmse_{k}", np.round(train_avg[k].cpu().numpy(), decimals=2), epoch)
        writer.add_scalar(f"eval_rmse_{k}", np.round(eval_avg[k].cpu().numpy(), decimals=2), epoch)
    writer.add_scalar("model_grad_norm", grad / len(calibrator.data["train"]), epoch)
    writer.add_scalar("learning_rate", calibrator.optimizer.param_groups[0]["lr"], epoch)
    writer.add_scalar("weight_norm", weight_norm, epoch)

    if "grape_phenology" in calibrator.config.DataConfig.dtype:
        best_avg = eval_avg[-1]
        if calibrator.best_cum_rmse > best_avg:
            calibrator.best_eval_loss = eval_loss
            calibrator.best_cum_rmse = best_avg
            calibrator.save_model(f"{fpath}", name="rnn_model_best.pt")
            calibrator.best_rmse = eval_avg
    elif "grape_coldhardiness" in calibrator.config.DataConfig.dtype:
        if calibrator.best_eval_loss > eval_loss:
            calibrator.best_eval_loss = eval_loss
            calibrator.save_model(f"{fpath}", name="rnn_model_best.pt")
    elif "wofost" in calibrator.config.DataConfig.dtype:
        if calibrator.best_eval_loss > eval_loss:
            calibrator.best_eval_loss = eval_loss
            calibrator.save_model(f"{fpath}", name="rnn_model_best.pt")

    # calibrator.save_model(f"{fpath}", name="rnn_model_current.pt")

    p_str = f"############### Epoch {epoch} ###############\n"
    p_str += f"Train loss: {train_loss}\n"
    p_str += f"Val loss: {eval_loss}\n"
    p_str += f"Model Grad Norm: {grad/len(calibrator.data['train'])}\n"
    if "grape_phenology" in calibrator.config.DataConfig.dtype:
        p_str += f"Train RMSE: {np.round(train_avg.cpu().numpy(),decimals=2)}\n"
        p_str += f"Val RMSE: {np.round(eval_avg.cpu().numpy(),decimals=2)}\n"
        p_str += f"Best Val RMSE: {np.round(calibrator.best_rmse.cpu().numpy(),decimals=2)}\n"

    print(p_str)


def log_error_training(
    calibrator: nn.Module,
    writer: SummaryWriter,
    fpath: str,
    epoch: int,
    train_loss: float,
    eval_loss: float,
    train_avg: float,
    eval_avg: float,
    grad: float,
) -> None:
    """
    Log training statistics and print to console
    """

    if hasattr(calibrator, "nn"):
        weight_norm = torch.sqrt(sum((p.data**2).sum() for p in calibrator.nn.parameters()))
    elif hasattr(calibrator, "finetuner"):
        weight_norm = torch.sqrt(sum((p.data**2).sum() for p in calibrator.finetuner.parameters()))

    writer.add_scalar("train_loss", train_loss, epoch)
    writer.add_scalar("eval_loss", eval_loss, epoch)
    for k in range(4):
        writer.add_scalar(f"train_rmse_{k}", np.round(train_avg[k].cpu().numpy(), decimals=2), epoch)
        writer.add_scalar(f"eval_rmse_{k}", np.round(eval_avg[k].cpu().numpy(), decimals=2), epoch)
    writer.add_scalar("model_grad_norm", grad / len(calibrator.data["train"]), epoch)
    writer.add_scalar("learning_rate", calibrator.optimizer.param_groups[0]["lr"], epoch)
    writer.add_scalar("weight_norm", weight_norm, epoch)

    if calibrator.best_eval_loss > eval_loss:
        calibrator.best_eval_loss = eval_loss
        calibrator.save_model(f"{fpath}", name="rnn_model_best.pt")

    # calibrator.save_model(f"{fpath}", name="rnn_model_current.pt")

    p_str = f"############### Epoch {epoch} ###############\n"
    p_str += f"Train loss: {train_loss}\n"
    p_str += f"Val loss: {eval_loss}\n"
    p_str += f"Model Grad Norm: {grad/len(calibrator.data['train'])}\n"

    print(p_str)
