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

### Loading a model 
See infer/inference_model.py for a generic interface. 

1. A valid configuration path must have (a) a `config.yaml` file, (2) a `rnn_model_best.pt` file for the NN weights, and (3) a `model_drange.pt` file for the normalization range of input data. 

2. To load a model, run: `python3 -m infer.inference_model --config _runs/DMC-MTL/Phenology/ParamMTL/All/dmc_mtl__1746932177` or alternatively call ```model = utils.load_inference_model_from_config(config_folder_fpath)```

### Performing model inference (daily predictions, forecasts, rollouts)


