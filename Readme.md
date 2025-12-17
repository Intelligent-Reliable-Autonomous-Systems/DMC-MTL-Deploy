## DMC-MTL: Hybrid Recurrent Models for Accurate Crop State Predictions
Developed by Will Solow and Sandhya Saisubramanian
Oregon State University, Corvallis, OR

Correspondance to Will Solow, soloww@oregonstate.edu 

### Installation

1. Create a virtual environment 
* `conda create -n dmc-deploy python=3.12`
* `conda activate dmc-deploy`

2. Clone this repository
* `https://github.com/Intelligent-Reliable-Autonomous-Systems/DMC-MTL-Deploy`

3. Install dependencies with pip
* `pip install -e model_engine`
* `pip install matplotlib requests tensorboard` 

### Running experiments
1. See available data in _data/processed_data/
2. Configure train.yaml file in _train_configs/
3. Run experiment with `python3 -m trainers.train_model --config train.yaml`