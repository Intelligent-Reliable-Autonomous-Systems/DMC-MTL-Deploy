"""
train_model.py

Entry interface for DMC-MTL inference.

Written by Will Solow, 2025
"""

import argparse
import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


def select_df(config, df_list:list[pd.DataFrame], i: int=-1) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Helper function for demonstration of selecting DataFrame
    """
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

    data, dates, cultivars, true_output = select_df(config, df_list, -5)
    cultivars = np.array(["Cabernet Sauvignon"])
    print(df_list[0])
    dmc_model.reset(1)
    output_arr = []
    for i in range(data.shape[0]):
        output = dmc_model.predict(data[i], dates[i:i+1], cultivars)
        output_arr.append(output)

    
    plt.title("Model Predictions vs True Observations")
    plt.plot(output_arr.flatten(), label="Model Prediction")
    plt.plot(true_output.flatten(), label="Model Prediction")
    plt.legend()
    plt.show()

    
    data, dates, cultivars, true_output = select_df(config, df_list, -1)
    drange = dmc_model.drange.cpu().numpy()[2:]
    noise_scale= drange[:,1] - drange[:,0]
    # Add large positive noise to visualize forecasts 
    forecast_data = data+np.abs(0.7*noise_scale*np.random.normal(size=(data.shape[0], noise_scale.shape[0])))

    dmc_model.reset(1)
    output_arr = []
    forecast_arr = []
    forecast_range = []
    forecast_step = 14
    for i in range(data.shape[0]):
        output = dmc_model.predict(data[i], dates[i:i+1], cultivars)
        if i % forecast_step == 0: 
            forecast = dmc_model.forecast(forecast_data[i:i+forecast_step], dates[i:i+forecast_step], cultivars)
            forecast_range.append(np.arange(i, i+forecast_step))
            forecast_arr.append(forecast)
        output_arr.append(output)
    output_arr = np.array(output_arr)
        
    plt.plot(true_output.flatten(), label="True Observations", c='r')
    # Plot 2 week forecast every two weeks
    for i in range(0, len(forecast_arr)):
        if i == 0: # Cheap hack to get the label for plt.legend to only print onece
            plt.plot(forecast_range[i], forecast_arr[i].flatten(), c='g', label="2 Week Forecast",alpha=0.6)
        else: 
            plt.plot(forecast_range[i], forecast_arr[i].flatten(), c='g')
    plt.plot(output_arr.flatten(), label="Model Predictions", c='b', alpha=0.6)
    plt.legend()
    plt.title("Model Predictions with 2 Week Forecasts")
    plt.show()


if __name__ == "__main__":
    main()
