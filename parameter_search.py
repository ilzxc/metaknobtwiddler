"""Performs bayesian parameter optimization using parameters described in
params.py file and data in inputted csv file"""
from __future__ import division
import os
import argparse
import numpy as np
import neural_networks
import bayesian_parameter_optimization as bpo
from params import nnet_params, hyperparameter_space


def set_trace():
    from IPython.core.debugger import Pdb
    import sys
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

RESULTS_PATH = 'results/'

if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv_path", type=str,
        help="Path to CSV file with distance(first col) and features")
    args = parser.parse_args()

    # Construct paths
    trial_directory = os.path.join(RESULTS_PATH, 'parameter_trials')
    model_directory = os.path.join(RESULTS_PATH, 'model')

    model = 'general'
    if model == 'general':
        # Load data given csv file, statistics are stored in model
        data = np.loadtxt(args.csv_path, dtype=object, delimiter=",")

        # Run parameter optimization forever
        bpo.parameter_search(data,
                             nnet_params['general'],
                             hyperparameter_space['general_network'],
                             trial_directory,
                             model_directory,
                             neural_networks.train,
                             model)
