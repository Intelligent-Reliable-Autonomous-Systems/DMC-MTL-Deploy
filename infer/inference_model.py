"""
train_model.py

Entry interface for DMC-MTL inference.

Written by Will Solow, 2025
"""

import argparse
import utils
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import torch

def select_df(config, df_list:list[pd.DataFrame], i: int=-1) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = df_list[i]
    cols = [c for c in config.PConfig.input_vars if c != "DAY"]
    data = df.loc[:,cols].to_numpy()
    dates = df.loc[:,"DAY"].to_numpy(dtype="datetime64[D]")
    cultivars = np.array([df.loc[:,"CULTIVAR"].to_numpy()[0]])
    true_output = df.loc[:,config.PConfig.output_vars].to_numpy()

    return data, dates, cultivars, true_output 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, type=str, help="Path to Config")
    parser.add_argument("--pt_file_name", default="rnn_model_best.pt", type=str, help=".pt file name")
    args = parser.parse_args()

    dmc_model, config = utils.load_inference_model_from_config(args)

    df_list = utils.load_data_from_config(config)

    data, dates, cultivars, true_output = select_df(config, df_list, -1)

    dmc_model.reset(1)

    output_arr = []
    param_arr = []
    for i in range(data.shape[0]):
        output, params = dmc_model.predict(data[i], dates[i:i+1], cultivars)
        output_arr.append(output)
        param_arr.append(params)

    output = np.array(output_arr)
    parr1 = torch.vstack(param_arr)

    
    data, dates, cultivars, true_output = select_df(config, df_list, -1)

    dmc_model.reset(1)

    output_arr = []
    param_arr = []

    for i in range(data.shape[0]):
        output, params = dmc_model.predict(data[i], dates[i:i+1], cultivars)
        forecast = dmc_model.forecast(data[i:i+2], dates[i:i+2], cultivars)
        output_arr.append(output)
        param_arr.append(params)



if __name__ == "__main__":
    main()
