#!/usr/bin/python
"""Performs bayesian parameter optimization using parameters described in
params.py file and data in inputted csv file"""
from __future__ import division
import os
import argparse
import numpy as np
import neural_networks
import bayesian_parameter_optimization as bpo
from params import nnet_params, hyperparameter_space
from params import TRIAL_DIRECTORY, MODEL_DIRECTORY


def set_trace():
    from IPython.core.debugger import Pdb
    import sys
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv_path", type=str,
        help="Path to CSV file with distance(first col) and features")

    parser.add_argument(
        "model_name", type=str, help="Name of the model to be trained")
    args = parser.parse_args()

    # Load data given csv file, statistics are stored in model
    data = np.loadtxt(args.csv_path, dtype='float32', delimiter=",")

    # Run parameter optimization forever
    bpo.parameter_search(data,
                         nnet_params[args.model_name],
                         hyperparameter_space['general_network'],
                         os.path.join(TRIAL_DIRECTORY, args.model_name),
                         MODEL_DIRECTORY,
                         neural_networks.train,
                         args.model_name)
