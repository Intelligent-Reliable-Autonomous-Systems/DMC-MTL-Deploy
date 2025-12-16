"""
train_model.py

Entry interface for RNN training.

Written by Will Solow, 2025
"""

import argparse
import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, type=str, help="Path to Config")
    parser.add_argument("--seed", default=0, type=int, help="Seed of Experiment")
    parser.add_argument("--cultivar", default=None, type=str, help="Cultivar Type")
    args = parser.parse_args()

    config, data = utils.load_config_data(args)

    calibrator = utils.load_model_from_config(config, data)

    calibrator.optimize()


if __name__ == "__main__":
    main()
