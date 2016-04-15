from __future__ import division
import lasagne

# random number seed
rand_num_seed = 1
n_params = 1

# neural network parameter not to be explored with bayesian parameter estimation
nnet_params = {
    'general': {'n_layers': 4,
                'batch_size': 16,
                'epoch_size': 128,
                'widths': [None, 1024, 1024, n_params],
                'non_linearities': (None,
                                    lasagne.nonlinearities.rectify,
                                    lasagne.nonlinearities.rectify,
                                    lasagne.nonlinearities.softmax),
                'update_func': lasagne.updates.adadelta
                },

}

# hyperparameter space to be explored using bayesian parameter optimization
hyperparameter_space = {
    'general_network': {
        'momentum': {'type': 'float', 'min': 0., 'max': 1.},
        'dropout': {'type': 'int', 'min': 0, 'max': 1},
        'learning_rate': {'type': 'float', 'min': .000001, 'max': .1},
        'network': {'type': 'enum', 'options': ['general_network']}
    },
}
