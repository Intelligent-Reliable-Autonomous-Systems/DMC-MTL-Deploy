"""
compile_runs.py

Compiles RNN data across multiple runs and takes the average RMSE,
saving data to file

Written by Will Solow, 2025
"""

import os
import numpy as np
import pickle as pkl
import argparse

import utils
from model_engine.util import CULTIVARS

from plotters.plot_utils import compute_total_RMSE, gen_all_data_and_plot
from plotters.plotting_functions import compute_rmse_plot


def main():

    argparser = argparse.ArgumentParser(description="Plotting script for model")
    argparser.add_argument("--start_dir", type=str, default="_runs/")
    argparser.add_argument("--break_early", action="store_true", help="If to break early when making data")
    argparser.add_argument("--save", action="store_true", help="If to save the plots")
    argparser.add_argument("--config", type=str, default="", help="To be filled in at runtime")
    argparser.add_argument("--prefix", type=str, default=None, help="Prefix of files to search for")
    argparser.add_argument("--num_runs", type=int, default=5)
    argparser.add_argument("--data_fname", type=str, default="rmse.txt")
    argparser.add_argument("--synth_test", type=str, default=None, help="Prefix of synthetic testing data")
    np.set_printoptions(precision=2)
    args = argparser.parse_args()

    config_dirs = utils.find_config_yaml_dirs(args.start_dir)

    # Setup total storage
    all_avg_pheno = np.zeros((8, args.num_runs))
    all_avg_ch = np.zeros((2, args.num_runs))
    all_cultivar_avg_pheno = np.zeros((len(CULTIVARS["grape_phenology_"]), 8, args.num_runs))
    all_cultivar_avg_ch = np.zeros((len(CULTIVARS["grape_coldhardiness_"]), 2, args.num_runs))
    for i, config in enumerate(config_dirs):
        print(config)
        if args.prefix is not None:
            if args.prefix not in config:
                continue

        args.config = f"{args.start_dir}/{config}"
        fpath = f"{os.getcwd()}/{args.config}"

        config, data, fpath = utils.load_config_data_fpath(args)
        args.cultivar = config.DataConfig.cultivar
        config.DConfig.batch_size = 128
        calibrator = utils.load_model_from_config(config, data)

        calibrator.load_model(f"{fpath}", name="rnn_model_best.pt")
        calibrator.eval()

        # Setup per run storage
        true_data = [[], [], []]
        output_data = [[], [], []]
        true_cultivar_data = [[[], [], []] for _ in range(calibrator.num_cultivars)]
        output_cultivar_data = [[[], [], []] for _ in range(calibrator.num_cultivars)]
        # Generate all data
        [
            gen_all_data_and_plot(
                config,
                fpath,
                args,
                calibrator,
                true_data,
                output_data,
                true_cultivar_data,
                output_cultivar_data,
                name=n,
            )
            for n in ["train", "test"]
        ]

        # Store data for averaging
        if "grape_phenology" in config.DataConfig.dtype:
            train_avg, _ = compute_rmse_plot(config, true_data[0], output_data[0], fpath, save=False)
            test_avg, _ = compute_rmse_plot(config, true_data[1], output_data[1], fpath, name="test", save=False)
            all_avg_pheno[:, i] = np.concatenate(
                (
                    train_avg[1:-1],
                    [np.sum(train_avg[1:-1])],
                    test_avg[1:-1],
                    [np.sum(test_avg[1:-1])],
                )
            )
        elif "grape_coldhardiness" in config.DataConfig.dtype:
            total_rmse, _ = compute_total_RMSE(true_data[0], output_data[0])
            test_total_rmse, _ = compute_total_RMSE(true_data[1], output_data[1])
            all_avg_ch[:, i] = np.array([total_rmse, test_total_rmse])

        for k in range(calibrator.num_cultivars):
            if len(true_cultivar_data[k][0]) == 0 and len(true_cultivar_data[k][1]) == 0:
                continue
            if "grape_phenology" in config.DataConfig.dtype:
                cultivar_train_avg_pheno, _ = compute_rmse_plot(
                    config,
                    true_cultivar_data[k][0],
                    output_cultivar_data[k][0],
                    fpath,
                    save=False,
                )
                cultivar_test_avg_pheno, _ = compute_rmse_plot(
                    config,
                    true_cultivar_data[k][1],
                    output_cultivar_data[k][1],
                    fpath,
                    name="test",
                    save=False,
                )
                all_cultivar_avg_pheno[k, :, i] = np.concatenate(
                    (
                        cultivar_train_avg_pheno[1:-1],
                        [np.sum(cultivar_train_avg_pheno[1:-1])],
                        cultivar_test_avg_pheno[1:-1],
                        [np.sum(cultivar_test_avg_pheno[1:-1])],
                    )
                )
            elif "grape_coldhardiness" in config.DataConfig.dtype:
                cultivar_train_rmse, _ = compute_total_RMSE(true_cultivar_data[k][0], output_cultivar_data[k][0])
                cultivar_test_rmse, _ = compute_total_RMSE(true_cultivar_data[k][1], output_cultivar_data[k][1])
                all_cultivar_avg_ch[k, :, i] = np.array([cultivar_train_rmse, cultivar_test_rmse])

    # Save Data
    prefix = args.synth_test + "_" if args.synth_test is not None else ""
    if "grape_phenology" in config.DataConfig.dtype:
        with open(f"{args.start_dir}/{prefix}results_agg_cultivars.pkl", "wb") as f:
            pkl.dump(all_avg_pheno, f)
        f.close()
        with open(f"{args.start_dir}/{prefix}results_per_cultivars.pkl", "wb") as f:
            pkl.dump(all_cultivar_avg_pheno, f)
        f.close()
    elif "grape_coldhardiness" in config.DataConfig.dtype:
        with open(f"{args.start_dir}/{prefix}results_agg_cultivars.pkl", "wb") as f:
            pkl.dump(all_avg_ch, f)
        f.close()
        with open(f"{args.start_dir}/{prefix}results_per_cultivars.pkl", "wb") as f:
            pkl.dump(all_cultivar_avg_ch, f)
        f.close()


if __name__ == "__main__":
    main()
