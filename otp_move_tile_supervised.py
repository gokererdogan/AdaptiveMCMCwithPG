"""
Learning data-driven proposals through reinforcement learning

Supervised training of a move tile proposal function for occluding tiles problem.

17 Jan. 2017
https://github.com/gokererdogan
"""
import numpy as np

import theano
import theano.tensor as T
import lasagne

from gmllib.helpers import progress_bar
import occluding_tiles_problem as otp

target_distribution = otp.OccludingTilesDistribution(ll_sd=1.0)


def get_batch(size=64):
    x = np.zeros((size, 1, otp.IMG_SIZE[0], otp.IMG_SIZE[1]), dtype=theano.config.floatX)
    y = np.zeros((size, 4), dtype=theano.config.floatX)
    for i in range(size):
        target = target_distribution.initial_x()
        current = target_distribution.initial_x()
        tile_ix = np.random.randint(6)
        change = target[tile_ix*4:(tile_ix+1)*4] - current[tile_ix*4:(tile_ix+1)*4]

        # theta is periodic
        if 12.0 - np.abs(change[-1]) < np.abs(change[-1]):
            if change[-1] < 0.0:
                change[-1] = 12.0 - np.abs(change[-1])
            else:
                change[-1] = -12.0 + np.abs(change[-1])

        x[i, 0] = (otp.OccludingTilesDistribution.render(current)[:, :, tile_ix] -
                   otp.OccludingTilesDistribution.render(target)[:, :, tile_ix]).astype(theano.config.floatX)
        y[i] = change
    return x, y


def train(epoch_count, batch_size, batches_per_epoch, batches_per_test_epoch):
    train_loss = np.zeros(epoch_count)
    test_loss = np.zeros(epoch_count)
    for e in range(epoch_count):
        # train
        epoch_loss = 0.0
        for b in range(batches_per_epoch):
            progress_bar(b+1, batches_per_epoch, update_freq=batches_per_epoch/100 or 1)
            tx, ty = get_batch(batch_size)
            l = train_fn(tx, ty)
            if np.isnan(l):
                raise RuntimeError("Nan!!!")
            epoch_loss += l
        epoch_loss /= batches_per_epoch
        train_loss[e] = epoch_loss
        print "Epoch {0:d}, avg. training loss: {1:f}".format(e+1, epoch_loss)

        # test
        epoch_loss = np.zeros(4)
        for b in range(batches_per_test_epoch):
            progress_bar(b+1, batches_per_test_epoch, update_freq=batches_per_test_epoch/100 or 1)
            tx, ty = get_batch(batch_size)
            l = test_fn(tx, ty)
            epoch_loss += l
        epoch_loss /= batches_per_test_epoch
        test_loss[e] = np.mean(epoch_loss)
        print "Epoch {0:d}, loss: {1}, \n\tavg. loss: {2}".format(e+1, epoch_loss, np.mean(epoch_loss))

    return train_loss, test_loss


def nn1():
    nn = lasagne.layers.InputLayer(shape=(None, 1, 50, 50))
    input = nn.input_var
    nn = lasagne.layers.DenseLayer(incoming=nn, num_units=4, nonlinearity=lasagne.nonlinearities.linear)
    return input, nn


def nn2():
    nn = lasagne.layers.InputLayer(shape=(None, 1, 50, 50))
    input = nn.input_var
    nn = lasagne.layers.DenseLayer(incoming=nn, num_units=200, nonlinearity=lasagne.nonlinearities.rectify)
    nn = lasagne.layers.DenseLayer(incoming=nn, num_units=4, nonlinearity=lasagne.nonlinearities.linear)
    return input, nn


def nn3():
    nn = lasagne.layers.InputLayer(shape=(None, 1, 50, 50))
    input = nn.input_var
    nn = lasagne.layers.Conv2DLayer(incoming=nn, num_filters=10, filter_size=(5, 5), stride=(2, 2),
                                    nonlinearity=lasagne.nonlinearities.rectify)
    nn = lasagne.layers.MaxPool2DLayer(incoming=nn, pool_size=(2, 2))
    nn = lasagne.layers.DenseLayer(incoming=nn, num_units=4, nonlinearity=lasagne.nonlinearities.linear)
    return input, nn


def nn4():
    nn = lasagne.layers.InputLayer(shape=(None, 1, 50, 50))
    input = nn.input_var
    nn = lasagne.layers.Conv2DLayer(incoming=nn, num_filters=10, filter_size=(2, 2), stride=(1, 1),
                                    nonlinearity=lasagne.nonlinearities.rectify)
    nn = lasagne.layers.MaxPool2DLayer(incoming=nn, pool_size=(2, 2))
    nn = lasagne.layers.DenseLayer(incoming=nn, num_units=4, nonlinearity=lasagne.nonlinearities.linear)
    return input, nn


def nn5():
    nn = lasagne.layers.InputLayer(shape=(None, 1, 50, 50))
    input = nn.input_var
    nn = lasagne.layers.Conv2DLayer(incoming=nn, num_filters=10, filter_size=(5, 5), stride=(2, 2),
                                    nonlinearity=lasagne.nonlinearities.rectify)
    nn = lasagne.layers.MaxPool2DLayer(incoming=nn, pool_size=(2, 2))
    nn = lasagne.layers.Conv2DLayer(incoming=nn, num_filters=40, filter_size=(3, 3), stride=(2, 2),
                                    nonlinearity=lasagne.nonlinearities.rectify)
    nn = lasagne.layers.MaxPool2DLayer(incoming=nn, pool_size=(2, 2))
    nn = lasagne.layers.DenseLayer(incoming=nn, num_units=4, nonlinearity=lasagne.nonlinearities.linear)
    return input, nn


def nn6():
    nn = lasagne.layers.InputLayer(shape=(None, 1, 50, 50))
    input = nn.input_var
    nn = lasagne.layers.Conv2DLayer(incoming=nn, num_filters=50, filter_size=(2, 2), stride=(1, 1),
                                    nonlinearity=lasagne.nonlinearities.rectify)
    nn = lasagne.layers.MaxPool2DLayer(incoming=nn, pool_size=(2, 2))
    nn = lasagne.layers.DenseLayer(incoming=nn, num_units=4, nonlinearity=lasagne.nonlinearities.linear)
    return input, nn


def nn7():
    nn = lasagne.layers.InputLayer(shape=(None, 1, 50, 50))
    input = nn.input_var
    nn = lasagne.layers.Pool2DLayer(incoming=nn, pool_size=(2, 2), mode='average_exc_pad')
    nn = lasagne.layers.DenseLayer(incoming=nn, num_units=100, nonlinearity=lasagne.nonlinearities.rectify)
    nn = lasagne.layers.DenseLayer(incoming=nn, num_units=4, nonlinearity=lasagne.nonlinearities.linear)
    return input, nn


def nn8():
    nn = lasagne.layers.InputLayer(shape=(None, 1, 50, 50))
    input = nn.input_var
    nn = lasagne.layers.Conv2DLayer(incoming=nn, num_filters=50, filter_size=(1, 1), stride=(1, 1),
                                    nonlinearity=lasagne.nonlinearities.rectify)
    nn = lasagne.layers.MaxPool2DLayer(incoming=nn, pool_size=(2, 2))
    nn = lasagne.layers.DenseLayer(incoming=nn, num_units=4, nonlinearity=lasagne.nonlinearities.linear)
    return input, nn


def nn9():
    nn = lasagne.layers.InputLayer(shape=(None, 1, 50, 50))
    input = nn.input_var
    nn = lasagne.layers.Conv2DLayer(incoming=nn, num_filters=50, filter_size=(3, 3), stride=(1, 1),
                                    nonlinearity=lasagne.nonlinearities.rectify)
    nn = lasagne.layers.MaxPool2DLayer(incoming=nn, pool_size=(2, 2))
    nn = lasagne.layers.DenseLayer(incoming=nn, num_units=4, nonlinearity=lasagne.nonlinearities.linear)
    return input, nn


def nn10():
    nn = lasagne.layers.InputLayer(shape=(None, 1, 50, 50))
    input = nn.input_var
    nn = lasagne.layers.Conv2DLayer(incoming=nn, num_filters=25, filter_size=(3, 3), stride=(1, 1),
                                    nonlinearity=lasagne.nonlinearities.rectify)
    nn = lasagne.layers.MaxPool2DLayer(incoming=nn, pool_size=(2, 2))
    nn = lasagne.layers.DenseLayer(incoming=nn, num_units=4, nonlinearity=lasagne.nonlinearities.linear)
    return input, nn


def nn11():
    nn = lasagne.layers.InputLayer(shape=(None, 1, 50, 50))
    input = nn.input_var
    nn = lasagne.layers.Conv2DLayer(incoming=nn, num_filters=50, filter_size=(3, 3), stride=(1, 1),
                                    nonlinearity=lasagne.nonlinearities.rectify)
    nn = lasagne.layers.MaxPool2DLayer(incoming=nn, pool_size=(4, 4))
    nn = lasagne.layers.DenseLayer(incoming=nn, num_units=4, nonlinearity=lasagne.nonlinearities.linear)
    return input, nn

if __name__ == "__main__":
    nns = [nn1, nn2, nn3, nn4, nn5, nn6, nn7, nn8, nn9, nn10, nn11]

    epoch_count = 100
    batches_per_epoch = 50
    batches_per_test_epoch = 20
    batch_size = 64

    for nn_i in [10]:
        np.random.seed(4031)
        input, nn = nns[nn_i]()
        output = lasagne.layers.get_output(nn)
        y = T.matrix('y')
        loss = lasagne.objectives.aggregate(lasagne.objectives.squared_error(output, y))
        elemwise_loss = T.mean(lasagne.objectives.squared_error(output, y), axis=0)
        params = lasagne.layers.get_all_params(nn)
        updates = lasagne.updates.adam(loss, params, learning_rate=0.01)
        train_fn = theano.function([input, y], loss, updates=updates)
        test_fn = theano.function([input, y], elemwise_loss)
        forward = theano.function([input], output)

        trl, tsl = train(epoch_count, batch_size, batches_per_epoch, batches_per_test_epoch)
        np.save('train_loss_{0}.npy'.format(nn_i), trl)
        np.save('test_loss_{0}.npy'.format(nn_i), tsl)

