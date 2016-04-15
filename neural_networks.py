import numpy as np
import theano
from theano import tensor as T
import lasagne
import matplotlib.pylab as plt
import seaborn


def set_trace():
    from IPython.core.debugger import Pdb
    import sys
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)


def get_next_batch(inputs, targets, batch_size, n_iters):
    for _ in range(n_iters):
        excerpt = np.random.permutation(len(inputs))[:batch_size]
        yield inputs[excerpt], targets[excerpt]


def train(data, layers, updates_fn, batch_size=64, epoch_size=128,
          initial_patience=1000, improvement_threshold=0.99,
          patience_increase=5, max_iter=100000):

    # specify input and target theano data types
    input_var = T.fvector('inputs')
    target_var = T.matrix('targets')

    # create a cost expression for training
    prediction = lasagne.layers.get_output(layers, input_var)
    cost = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    cost = cost.mean()

    # create parameter update expressions for training
    params = lasagne.layers.get_all_params(layers, trainable=True)
    updates = updates_fn(cost, params)

    # compile functions for performing training step and returning
    # corresponding training cost
    train_fn = theano.function(inputs=[input_var, target_var],
                               outputs=cost,
                               updates=updates)

    # create cost expression for validation
    # deterministic forward pass to disable droupout layers
    val_prediction = lasagne.layers.get_output(layers, input_var,
                                               deterministic=True)
    val_cost = lasagne.objectives.categorical_crossentropy(
        val_prediction, target_var)
    val_cost = val_cost.mean()

    # compile a function to compute the validation cost and objective function
    validate_fn = theano.function(inputs=[input_var, target_var],
                                  outputs=val_cost)

    # build a prediction function to manually evaluate output
    val_output = theano.function([input_var], val_prediction)

    # create data iterators
    train_data_iter = get_next_batch(data['train']['distance'],
                                     data['train']['parameters'],
                                     batch_size, max_iter)

    patience = initial_patience
    current_val_cost = np.inf
    train_cost = 0.0

    for n, (x_batch, y_batch) in enumerate(train_data_iter):
        train_cost += train_fn(x_batch, y_batch)

        # Stop training if NaN is encountered
        if not np.isfinite(train_cost):
            print 'Bad training er {} at iteration {}'.format(train_cost, n)
            break

        if n and not (n % epoch_size):
            epoch_result = {'iteration': n,
                            'train_cost': train_cost / float(epoch_size),
                            'validate_cost': 0.0,
                            'validate_objective': 0.0}

            # compute validation cost and objective
            cost = np.float(validate_fn(data['validate']['distance'],
                                        data['validate']['parameters']))

            epoch_result['validate_cost'] = float(cost)
            epoch_result['validate_objective'] = float(cost)

            # Test whether this validate cost is the new smallest
            if epoch_result['validate_cost'] < current_val_cost:
                # To update patience, we must be smaller than
                # improvement_threshold*(previous lowest validation cost)
                patience_cost = improvement_threshold*current_val_cost
                if epoch_result['validate_cost'] < patience_cost:
                    # Increase patience by the supplied about
                    patience += epoch_size*patience_increase
                # Even if we didn't increase patience, update lowest valid cost
                current_val_cost = epoch_result['validate_cost']
            # Store patience after this epoch
            epoch_result['patience'] = patience

            if n > patience:
                break

            yield epoch_result


def build_general_network(input_shape, input_mean, input_std, n_layers, widths,
                          non_linearities, drop_out=True):
    """
    Parameters
    ----------
    input_shape : tuple of int or None (batchsize, rows, cols)
        Shape of the input. Any element can be set to None to indicate that
        dimension is not fixed at compile time
    """

    # GlorotUniform is the default mechanism for initializing weights
    for i in range(n_layers):
        if i == 0:  # input layer
            layers = lasagne.layers.InputLayer(shape=input_shape)
            layers.append(lasagne.layers.standardize(
                layers[-1], input_mean, input_std, shared_axes=(0, 2)))
        else:  # hidden and output layers
            layers.append(
                lasagne.layers.DenseLayer(layers[-1],
                                          num_units=widths[i],
                                          nonlinearity=non_linearities[i])
            if drop_out and i < n_layers-1:  # output layer has no dropout
                layers.append(lasagne.layers.DropoutLayer(layers[-1], p=0.5)

    return layers
