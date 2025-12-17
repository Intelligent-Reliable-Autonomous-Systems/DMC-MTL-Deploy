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

    dmc_model, config = utils.load_inference_model_from_config(args)

    df = utils.load_data_from_config(config)[-1]
    
    cols = [c for c in config.PConfig.input_vars if c != "DAY"]
    data = df.loc[:,cols].to_numpy()
    dates = df.loc[:,"DAY"].to_numpy(dtype="datetime64[D]")
    cultivars = np.array([df.loc[:,"CULTIVAR"].to_numpy()[0]])
    true_output = df.loc[:,config.PConfig.output_vars].to_numpy()
    
    dmc_model.reset(1)
    output = dmc_model.predict(data, dates, cultivars)


if __name__ == "__main__":
    main()
