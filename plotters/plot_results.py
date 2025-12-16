"""
plot_results.py

Main plotting function for RNN-based methods
Handles the generation and processing of data on a per batch basis
and then calls correct plot functions

Written by Will Solow, 2025
"""

import os
import numpy as np
import argparse

import utils

from plotters.plot_utils import gen_all_data_and_plot
from plotters.plotting_functions import plot_loss, plot_stats


def main():

    argparser = argparse.ArgumentParser(description="Plotting script for model")
    argparser.add_argument("--config", type=str, required=True, help="Path to the config file")
    argparser.add_argument("--break_early", action="store_false", help="If to break early when making data")
    argparser.add_argument("--save", action="store_false", help="If to save the plots")
    argparser.add_argument("--fname", type=str, default="data.txt", help="File to save data to")
    argparser.add_argument("--rnn_name", type=str, default="rnn_model_best.pt")
    np.set_printoptions(precision=2)

    args = argparser.parse_args()
    fpath = f"{os.getcwd()}/{args.config}"

    config, data, fpath = utils.load_config_data_fpath(args)

    # Plot tensorboard outputs
    try:
        plot_loss(fpath, config)
        plot_stats(fpath, config)
    except:
        pass

    calibrator = utils.load_model_from_config(config, data)
    calibrator.load_model(f"{fpath}", name=args.rnn_name)
    calibrator.eval()

    # Setup Storage
    true_data = [[], [], []]
    output_data = [[], [], []]
    true_cultivar_data = [[[], [], []] for _ in range(calibrator.num_cultivars)] 
    output_cultivar_data = [[[], [], []] for _ in range(calibrator.num_cultivars)] 

    # Generate all data
    [gen_all_data_and_plot(
        config,
        fpath,
        args,
        calibrator,
        true_data,
        output_data,
        true_cultivar_data,
        output_cultivar_data,
        name=t,
    )
    for t in ["train", "test"]]



if __name__ == "__main__":
    main()
