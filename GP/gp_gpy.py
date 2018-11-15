"""Convenience functions for GPs using GPy."""

import os
import numpy as np
import GPy

__gp_dir = r'./gps/'

def __gp_path(name: str, traj_n: int, seg_n: int):
    return __gp_dir + __gp_file_name(name, traj_n, seg_n)

def __gp_file_name(name: str, traj_n: int, seg_n: int):
    return name + '-' + str(traj_n) + '.' + str(seg_n) + '.npy'

def save(model: GPy.models.GPRegression, name: str, traj_n: int, seg_n: int):
    """
    Generate a path from provided name, traj, and seg, and store the model.
    KNOWN BUG: The priors of the model disappera when stored.
    """
    if not os.path.exists(__gp_dir):
        os.makedirs(__gp_dir)
    print(model)
    print(name)
    print(traj_n)
    print(seg_n)
    
    np.save(__gp_path(name, traj_n, seg_n), model.param_array)

def load(X: np.ndarray, Y: np.ndarray, name: str, traj_n: int, seg_n: int):
    """
    Load model stored for the provided name, traj and seg.
    KNOWN BUG: The priors of the model disappera when stored.
    """
    model = GPy.models.GPRegression(X, Y)
    model.update_model(False)
    model.initialize_parameter()
    model[:] = np.load(__gp_path(name, traj_n, seg_n))
    model.update_model(True)
    return model
