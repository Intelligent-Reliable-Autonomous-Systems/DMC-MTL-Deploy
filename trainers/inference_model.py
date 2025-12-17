"""
train_model.py

Entry interface for DMC-MTL inference.

Written by Will Solow, 2025
"""

import argparse
import utils
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, type=str, help="Path to Config")
    parser.add_argument("--pt_file_name", default="rnn_model_best.pt", type=str, help=".pt file name")
    args = parser.parse_args()

    config, fpath = utils.load_config_fpath(args)

    dmc_model = utils.load_inference_model_from_config(config, fpath, args.pt_file_name)

    dmc_model.infer(np.arange(15), np.array(["2023-01-01"], dtype=np.datetime64), np.array([1]))


if __name__ == "__main__":
    main()
