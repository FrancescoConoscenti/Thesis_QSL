import os
import pickle
import numpy as np
import re
import flax

class MockLog:
    def __init__(self, data):
        self.data = data

def load_log(folder):
    log_path = os.path.join(folder, "log.pkl")
    old_log_data = {}
    if os.path.exists(log_path):
        try:
            with open(log_path, 'rb') as f:
                old_log_data = pickle.load(f)
        except Exception as e:
            print(f"Could not load existing log: {e}")
    return old_log_data

def merge_log_data(old_data, new_data):
    if not old_data:
        return new_data
    if not new_data:
        return old_data
    
    merged = old_data.copy()
    for key, val in new_data.items():
        if key in merged:
            if isinstance(val, dict):
                merged[key] = merge_log_data(merged[key], val)
            elif hasattr(val, '__iter__') and not isinstance(val, str):
                 # Concatenate arrays/lists
                 if isinstance(merged[key], np.ndarray) or isinstance(val, np.ndarray):
                     merged[key] = np.concatenate((np.array(merged[key]), np.array(val)))
                 else:
                     merged[key] = list(merged[key]) + list(val)
        else:
            merged[key] = val
    return merged

def load_checkpoint(save_model, block_iter, save_every, vstate):
    start_block = 0
    if os.path.exists(save_model):
        model_files = [f for f in os.listdir(save_model) if f.startswith("model_") and f.endswith(".mpack")]
        indices = []
        for f in model_files:
            match = re.search(r"model_(\d+)\.mpack", f)
            if match:
                indices.append(int(match.group(1)))
        
        if indices:
            last_block = max(indices)
            if last_block <= block_iter:
                print(f"Resuming from block {last_block} (iteration {last_block * save_every})")
                with open(os.path.join(save_model, f"model_{last_block}.mpack"), 'rb') as file:
                    data = file.read()
                    try:
                        vstate = flax.serialization.from_bytes(vstate, data)
                    except KeyError:
                        vstate.variables = flax.serialization.from_bytes(vstate.variables, data)
                start_block = last_block
    return start_block, vstate