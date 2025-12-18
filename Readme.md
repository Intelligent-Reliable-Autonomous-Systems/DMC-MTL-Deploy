# DMC-MTL-Deploy: Dynamic Model Calibration via Multi-Task Learning
Developed by Will Solow and Sandhya Saisubramanian
Oregon State University, Corvallis, OR

Correspondance to Will Solow, soloww@oregonstate.edu 

This respository handles the deployment of hybrid phenology models on AgWeatherNet.

## Installation

1. Create a virtual environment 
* `conda create -n dmc-deploy python=3.12`
* `conda activate dmc-deploy`

2. Clone this repository
* `https://github.com/Intelligent-Reliable-Autonomous-Systems/DMC-MTL-Deploy`

3. Install dependencies with pip
* `pip install -e model_engine`
* `pip install matplotlib requests tensorboard` 

## Demonstration
See `infer/inference_model.py` for a demonstration. This can be run with `python3 -m infer.demo --config runs/DMC-MTL/Phenology/All/dmc_mtl__1` and will output two plots: 
1. A seasonal plot of phenology predictions from Jan 1 to October 7th with the true values
2. A seasonal plot of daily phenology predictions (with true weather data) and biweekly 14 day forecasts with noisy weather data. 

## General Use Instructions 

### Loading a model 
See infer/inference_model.py for a generic interface. 

1. A valid configuration path must have (a) a `config.yaml` file, (2) a `rnn_model_best.pt` file for the NN weights, and (3) a `model_drange.pt` file for the normalization range of input data. 

2. To load a model, run: `python3 -m infer.inference_model --config _runs/DMC-MTL/Phenology/All/dmc_mtl__1` or alternatively call ```model = utils.load_inference_model_from_config(config_folder_filepath)```

### Performing model inference (daily predictions, forecasts, rollouts)

See `infer/demo.py` for an example. The `DMC_Inference` model class exposes 3 interfaces:

1. `reset(batch_size)`. Must be called at the beginning of each season. Handles resetting the RNN + biophysical model. `batch_size` is the number of cultivars to be predicted for simultaneously. Ensure that batch_size in `config.yaml` is greater than or equal to this batch_size. The smaller the batch size, the less GPU resources/faster the model will run on CPU. If a GPU is available, it would be possible run daily predictions for all 31 cultivars simultaneously. 

2. `predict(weather_data, dates, cultivar)`. This is the main driver of the prediction model. It should be called daily with with the observed weather, or every t days with the previous t days of weather. Handles stepping the model forward by t days. 

    (a) Inputs: weather_data: a numpy ndarray with size [num_cultivars, num_days, num_weather_features]. Can be 1, 2 or 3-dimensional, if 1-dimensional, dates must also be of length 1. If         2-dimensional, first dimension must match either size of `dates` or `cultivars`. If 3-dimensional, first dimension must match cultivars, second dimension must match dates. This size checking is handled. 
                dates: a numpy ndarray of type np.datetime64 or list of strings in YYYY-MM-DD format. Must be 1-dimensional
                cultivars: a numpy ndarray of strings or ints. If strings, must correspond to cultivars in Supported Cultivars (below) or be integer in alphabetical list of cultivars. See `model_engine.util.py:CULTIVARS`. Must be 1-dimesional

    (b) Outputs: `phenology`: a numpy ndarray with size [num_cultivars, num_days, num_weather_features]. If 1 cultivar and 1 date is passed in, the output will be of size (1,1,1). The phenology is an integer in the range [0,4] with 0: Ecodormancy, 1: Bud Break, 2: Bloom, 3: Veraison (50%), 4: Ripe/Harvest. The first occurance of a new integer in the output corresponds to the onset of the phenological stage. 

3. `forecast(weather_data, dates, cultivar)`. This is the forecasting method of the prediction model. It should be called whenever a k day forecast is desired. It outputs a k-day forecast before resetting the model to the state it was in before forecast() was called. 

    (a) Inputs: `weather_data`: a numpy ndarray with size [num_cultivars, num_days, num_weather_features]. Can be 1, 2 or 3-dimensional, if 1-dimensional, dates must also be of length 1. If 2-dimensional, first dimension must match either size of `dates` or `cultivars`. If 3-dimensional, first dimension must match cultivars, second dimension must match dates. This size checking is handled. 
                `dates`: a numpy ndarray of type np.datetime64 or list of strings in YYYY-MM-DD format. Must be 1-dimensional
                `cultivars`: a numpy ndarray of strings or ints. If strings, must correspond to cultivars in Supported Cultivars (below) or be integer in alphabetical list of cultivars. See `model_engine.util.py:CULTIVARS`. Must be 1-dimesional

    (b) Outputs: `phenology forecast`: a numpy ndarray with size [num_cultivars, num_days, num_weather_features]. If 1 cultivar and 1 date is passed in, the output will be of size (1,1,1). The phenology is an integer in the range [0,4] with 0: Ecodormancy, 1: Bud Break, 2: Bloom, 3: Veraison (50%), 4: Ripe/Harvest. The first occurance of a new integer in the output corresponds to the onset of the phenological stage. 


### Model Inputs
The DMC-MTL base model requires access to 15 weather variables and the date in a YYYY-MM-DD format. These weather variables are: 
- Daily min, max, and average temperature in Celcius, 
- Daily rainfall in Centimeters (can convert from AWN by multplying by 2.54), 
- Daily solar irradiation in Joules (can convert from AWN by multiplying by 1e7), 
- Daylength in hours, 
- Daily min, max, and average relative humidity (%), 
- Daily min, max, and average dew point in degrees Celcius, 
- Daily wind speed and max wind speed in meters/second (can convert from AWN by multiplying by 0.44704),
- Daily evapotranpiration (Eto). Can be grabbed directly from AWN data. 

    The inputs must be in the order: Average Temperature, Min Temperature, Max Temperature, Rainfall, Solar Irradiation, Daylength, Min Relative Humidity, Average Relative Humidity, Max Relative Humidity, Min Dew Piont, Average Dew Point, Max Dew Point, Wind Speed, Max Wind Speed, Evapotranspiration. 

    See the `config.yaml` file for the loaded model to double check the required order. Ignore the "DAY" input - This indicates if the model requires a sine/cosine embedding of the date, which is handled in the predict() and forecast() methods. 

### Model Outputs
For both the `predict()` and `forecast()` methods, the output of the model will be one of [0, 1, 2, 3, 4] where:
    - 0: The grape is in Ecodormancy
    - 1: The grape is in Bud Break
    - 2: The grape is in Bloom
    - 3: The grape is in Veraison 50%
    - 4: The grape is in Ripe/Harvest

For forecasting purposes, the first occurance of a new integer corresponds to a new phenological stage being reached/the day of onset. For example, if the return of a forecast call with the dates [2025-05-01, 2025-05-02, 2025-05-03] returned [[[0,1,1]]], this means that the model forecasts that Bud Break will occur on 25-05-02. 

The Ripe/Harvest value can be ignored. The DMC-MTL model predicts the average day of onset like the Parker model. It does not predict the min and max days of onset as well (like the Zapata model). 

### Supported Cultivars
1. Phenology: Aligote, Alvarinho, Auxerrois, Barbera, Cabernet Franc, Cabernet Sauvignon, Chardonnay, Chenin Blanc, Concord, Durif, Gewurztraminer, Green Veltliner, Grenache, Lemberger, Malbec, Melon, Merlot, Mourvedre, Muscat Blanc, Nebbiolo, Petit Verdot, Pinot Blanc, Pinot Gris, Pinot Noir, Riesling, Sangiovese, Sauvignon Blanc, Semillon, Tempranillo, Viognier, Zinfandel


2. Cold Hardiness: Barbera, Cabernet Franc, Cabernet Sauvignon, Chardonnay, Chenin Blanc, Concord, Gewurztraminer, Grenache, Lemberger, Malbec, Merlot, Mourvedre, Nebbiolo, Pinot Gris, Riesling, Sangiovese, Sauvignon Blanc, Semillon, Syrah, Viognier, Zinfandel

