"""
Initial entry point for model engines and make data cache
"""

# Import first to avoid circular imports
from . import util
from . import models
import os

import pathlib

user_path = pathlib.Path(__file__).parent.resolve()

# Make .weather_data cache folder in the current working directory
weather_data_user_home = os.path.join(f"{user_path}/inputs", ".weather_data")
os.makedirs(weather_data_user_home, exist_ok=True)

# Make folder in .weather_data for weather data
meteo_cache_dir = os.path.join(weather_data_user_home, "meteo_cache")
os.makedirs(meteo_cache_dir, exist_ok=True)
