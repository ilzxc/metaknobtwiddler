#!/usr/bin/python

"""This function loads the best model trained so far and uses it to make
predictions on the csv dataset provided"""

import os
import argparse
import cPickle as pkl
import numpy as np
import theano
from theano import tensor as T
import lasagne
import deepdish
import neural_networks
from params import RESULTS_DIRECTORY
from params import nnet_params


def set_trace():
    from IPython.core.debugger import Pdb
    import sys
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv_path", type=str,
        help="Fullpath to CSV file with rows as 'distance, params'")
    parser.add_argument(
        "model_path", type=str,
        help="Fullpath to H5 file with model")
    args = parser.parse_args()

    model_preds = {}

    model_name = os.path.basename(args.model_path)
    model_name = model_name[:model_name.rfind('.')]
    print("\nExecuting prediction on test set \n{}").format(model_name)
    # Load data given csv file, statistics are stored in model
    data = np.loadtxt(args.csv_path, dtype=object, delimiter=",")

    # Build best model structure
    network = neural_networks.build_general_network(
        (nnet_params['batch_size'], data.shape[1]-1),  # first is target
        nnet_params['n_layers'],
        nnet_params['widths'],
        nnet_params['non_linearities'],
        drop_out=False)

    # Load best parameters on best model topology
    parameters = deepdish.io.load(args.model_path)
    lasagne.layers.set_all_param_values(network, parameters)

    # Set up neural network variables and functions for prediction
    input_var = T.fvector()
    target_var = T.fmatrix()
    prediction = lasagne.layers.get_output(
        network, input_var, deterministic=True)
    obj_fn = T.mean(T.neq(T.argmax(prediction, axis=1), target_var))

    validate_fn = theano.function(
        inputs=[input_var, target_var], outputs=[obj_fn])

    # Compute predictions. first column is target variable
    obj_val = validate_fn(data[:, 1:].astype(input_var.dtype),
                          data[:, 0].astype(target_var.dtype))
    model_preds[model_name] = obj_val
    print("{} error rate on test set: {}").format(model_name, obj_val)

    # Dump predictions dictionary
    pkl.dump(model_preds,
             open(os.path.join(RESULTS_DIRECTORY, 'results.np', 'wb')))
