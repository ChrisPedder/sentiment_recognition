import os
import argparse
import json
import numpy as np

from datetime import datetime

# Helper function to safely create directories
def safe_folder_create(path):
    if not os.path.exists(path):
        os.makedirs(path)

def walklevel(some_dir, level=1):
    """
    Same functionality as os.walk, but allows you to stop at a certain depth
    given by 'level'.
    """
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]


def create_model_training_folder(log_dir):
    # Check that top level log dir exists, if not, create it
    safe_folder_create(log_dir)

    # Next-level log dir based on date, if not already present, create it
    now = datetime.now()
    date = now.strftime("%d_%m_%Y")
    date_dir = os.path.join(log_dir, date)
    safe_folder_create(date_dir)

    # Lowest-level log dir based on numbering, if date_dir not empty,
    # check that the previous highest index was, and increment by one.
    last_index = 0
    if len(os.listdir(date_dir)) != 0:
        subfolder_list = [x[0] for x in walklevel(date_dir) if os.path.isdir(x[0])]
        last_index = max([int(x.split('_')[-1]) for x in subfolder_list[1:]])

    model_dir = os.path.join(date_dir, 'model_' + str(last_index + 1))
    safe_folder_create(model_dir)
    return model_dir

# Helper function for writing to JSON
def jsonify(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
        return int(obj)
    return obj

# Helper function to write an argparse namespace object to json
def write_config_to_json(config: argparse.Namespace, logs_dir: str) -> None:
    args_dict = vars(config)
    for key, value in args_dict.items():
        args_dict[key] = jsonify(value)
    json_path = os.path.join(logs_dir, 'config.json')
    with open(json_path, 'w') as f:
        json.dump(args_dict, f)
    print(f'Config file written to {json_path}')
