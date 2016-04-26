import numpy as np
import theano
from theano import tensor as T
import lasagne
import pylab as plt


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
    input_var = T.matrix('inputs')
    target_var = T.matrix('targets')

    # define Theano correlation function
    x = T.fmatrix('data_centered')
    y = T.fscalar('normalization')
    corr_f = (T.dot(x, x.T.conj()) / y) / T.sqrt(
        T.outer(T.diag(T.dot(x, x.T.conj()) / y),
                T.diag(T.dot(x, x.T.conj()) / y)))
    corr_t = theano.function([x, y], corr_f)

    # create a cost expression for training
    prediction = lasagne.layers.get_output(layers, input_var)
    mse = lasagne.objectives.squared_error(
        prediction, target_var).mean()
    set_trace()
    prediction_corr = corr_t(prediction, prediction.shape[1]-1)
    # target_corr = corr_t(target_var)
    # change for more appropriate distance measures
    # corr_mse = lasagne.objectives.squared_error(
    #     prediction_corr, target_corr).mean()

    cost = mse

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

    val_cost = lasagne.objectives.squared_error(
        val_prediction, target_var).sum()

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
        train_cost += train_fn(x_batch.reshape(len(x_batch), 1), y_batch)

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
            cost = np.float(validate_fn(
                data['validate']['distance'].reshape((
                    len(data['validate']['distance']), 1)),
                data['validate']['parameters']))

            outs = val_output(data['validate']['distance'].reshape((
                    len(data['validate']['distance']), 1)))

            if cost < 400:
                print '\ncorrelation'
                print np.corrcoef(np.column_stack((data['validate']['distance'],
                                                   outs)).T)

                plt.figure(figsize=(12, 8))
                plt.subplot(4, 1, 1)
                plt.title('Param1')
                plt.plot(data['validate']['parameters'][:, 0], 'g.')
                plt.plot(outs[:, 0], 'r.')

                plt.subplot(4, 1, 2)
                plt.title('Absolute Differences Param 1')
                plt.plot(data['validate']['parameters'][:, 0]-outs[:, 0], 'b.')

                plt.subplot(4, 1, 3)
                plt.title('Param2')
                plt.plot(data['validate']['parameters'][:, 1], 'g.')
                plt.plot(outs[:, 1], 'r.')

                plt.subplot(4, 1, 4)
                plt.title('Absolute Differences Param 1')
                plt.plot(data['validate']['parameters'][:, 1]-outs[:, 1], 'b.')

                plt.show()

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


def build_general_network(input_shape, n_layers, widths, non_linearities,
                          drop_out=True, input_mean=False, input_std=False):
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
            # if input_mean is not False and input_std is not False:
            #     layers = lasagne.layers.standardize(
            #      layers, input_mean, input_std, shared_axes=(0, 2))
        else:  # hidden and output layers
            layers = lasagne.layers.DenseLayer(
                layers,
                num_units=widths[i],
                nonlinearity=non_linearities[i])
            if drop_out and i < n_layers-1:  # output layer has no dropout
                layers = lasagne.layers.DropoutLayer(layers, p=0.5)

    return layers
