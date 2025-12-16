"""
load_results.py

Loads data from pickle and prints statistics

Written by Will Solow, 2025
"""

from pathlib import Path
import pickle
import argparse
from argparse import Namespace
import numpy as np

from model_engine.util import CULTIVARS


def load_named_pickles(folder_paths: list[str], target_name: str, args: Namespace):
    """
    Load all pickle files matching a given name in all subdirectories.
    """
    results = {}

    for root in folder_paths:
        root = Path(f"./{root}")
        for pkl_file in root.rglob(target_name):
            try:
                # Get relative subdirectory name
                subdir = "/".join(pkl_file.parent.parts[-4:])
                if "All" in pkl_file.parent.parts and args.stl:  # subdir
                    continue
                # print(subdir)
                print(pkl_file)
                with open(pkl_file, "rb") as f:
                    results[subdir] = pickle.load(f)
            except Exception as e:
                pass
    return results


def compute_str_stl_ndim2(mtl_arr: np.ndarray, mean: np.ndarray, std: np.ndarray):
    """Compute print string for when loading single STL model"""
    all_str = ""
    for i in range(mean.shape[0]):
        if (np.isnan(mean[i])).all():
            continue
        for j in range(mean.shape[1]):
            all_str += f"{mean[i,j]} +/- {std[i,j]}, "
        all_str += "\n"
    all_mean = np.round(np.nanmean(mtl_arr, axis=(0, -1)), decimals=2)
    all_std = np.round(np.nanstd(mtl_arr, axis=(0, -1)), decimals=2)
    for i in range(all_mean.shape[0]):
        all_str += f"{all_mean[i]} +/- {all_std[i]}, "
    return all_str


def compute_all_mean(mtl_arr: np.ndarray, start: int, end: int):
    """
    Compute all mean of array with start and ends to split stations
    """
    all_str = ""
    all_mean = np.round(np.nanmean(mtl_arr[start:end], axis=(0, 1, -1)), decimals=2).squeeze()
    all_std = np.round(np.nanstd(mtl_arr[start:end], axis=(0, 1, -1)), decimals=2).squeeze()
    all_str += "\n"
    for i in range(all_mean.shape[0]):
        all_str += f"{all_mean[i]} +/- {all_std[i]}, "
    return all_str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_dir", type=str, default="", help="Path to results .pkls")
    parser.add_argument("--prefix", type=str, help="prefix of results file", default="")
    parser.add_argument("--stl", action="store_true", help="If to toggle printing for STL variant")
    parser.add_argument("--station", action="store_true", help="If to toggle MTL printing variant by station")
    parser.add_argument("--site", action="store_true", help="If to toggle MTL printint variant by site")
    parser.add_argument("--cult", action="store_true", help="If to toggle MTL printint variant by cult")
    parser.add_argument("--per", action="store_false", help="If to load per cultivar or aggregate file")
    parser.add_argument("--per_cult", action="store_false", help="If to print per cultivar results")
    args = parser.parse_args()

    fpath = args.prefix + "results_agg_cultivars.pkl"
    fpath = fpath.replace("agg", "per") if args.per else fpath
    mtl_models = load_named_pickles([args.start_dir], fpath, args)

    sorted_keys = np.argsort(list(mtl_models.keys()))  # Reorder based on alphabetical
    mtl_arr = np.array(list(mtl_models.values()))[sorted_keys]
    mtl_arr = np.where(mtl_arr == 0, np.nan, mtl_arr)  # Replace 0.0s with nan
    # Take average over runs and cultivars
    if args.per:
        if args.per_cult:
            mean = np.round(np.nanmean(mtl_arr, axis=-1), decimals=2).squeeze()
            std = np.round(np.nanstd(mtl_arr, axis=-1), decimals=2).squeeze()
        else:
            mean = np.round(np.nanmean(mtl_arr, axis=(-3, -1)), decimals=2).squeeze()
            std = np.round(np.nanstd(mtl_arr, axis=(-3, -1)), decimals=2).squeeze()
    else:
        mean = np.round(np.nanmean(mtl_arr, axis=-1), decimals=2).squeeze()
        std = np.round(np.nanstd(mtl_arr, axis=-1), decimals=2).squeeze()

    print(mean.shape)
    if args.per_cult:
        all_str = compute_str_stl_ndim2(mtl_arr, mean, std)

    else:  # For when we are not loading individual cultivar data
        all_str = ""
        if mean.ndim == 5 and args.rtmc:
            all_str = compute_str_ndim5_rtmc(mtl_arr, mean, std)
        elif mean.ndim == 4:  # For when we have region/station/site
            for j in range(mean.shape[0]):
                for k in range(mean.shape[1]):
                    for l in range(mean.shape[2]):
                        if (np.isnan(mean[j, k, l])).all():
                            continue
                        for i in range(mean.shape[3]):
                            all_str += f"{mean[j,k,l,i]} +/- {std[j,k,l,i]}, "
                        all_str += "\n"
                    all_str += "\n"
                all_str += "\n"
        elif mean.ndim == 3:  # For when we have region/station
            for j in range(mean.shape[0]):
                for k in range(mean.shape[1]):
                    if (np.isnan(mean[j, k])).all():
                        continue
                    for i in range(mean.shape[2]):
                        all_str += f"{mean[j,k,i]} +/- {std[j,k,i]}, "
                    all_str += "\n"
                all_str += "\n"
        elif mean.ndim == 2 and args.stl:  # For when we have multiple stl models
            for j in range(mean.shape[0]):
                for i in range(mean.shape[1]):
                    all_str += f"{mean[j,i]} +/- {std[j,i]}, "
                all_str += "\n"
            all_mean = np.round(np.nanmean(mtl_arr, axis=(0, 1, -1)), decimals=2)
            all_std = np.round(np.nanstd(mtl_arr, axis=(0, 1, -1)), decimals=2)
            for i in range(all_mean.shape[0]):
                all_str += f"{all_mean[i]} +/- {all_std[i]}, "
        else:  # For when we have just train/test/val
            for i in range(mean.shape[0]):
                all_str += f"{mean[i]} +/- {std[i]}, "
    print(all_str)


if __name__ == "__main__":
    main()
