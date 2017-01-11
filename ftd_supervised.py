"""
Learning data-driven proposals through reinforcement learning

Supervised training of a proposal function for find the dot problem.

4 Jan. 2017
https://github.com/gokererdogan
"""
import numpy as np

import theano
import theano.tensor as T
import lasagne

from gmllib.helpers import progress_bar
import find_the_dot_problem as ftd


def get_batch(size=64):
    x = np.zeros((size, 1, ftd.IMG_SIZE), dtype=theano.config.floatX)
    y = np.zeros((size, 1), dtype=theano.config.floatX)
    for i in range(size):
        target = np.random.choice(ftd.XS)
        observed = np.random.choice(ftd.XS)
        x[i, 0] = (ftd.FindTheDotTargetDistribution.render(observed) -
                   ftd.FindTheDotTargetDistribution.render(target)).astype(theano.config.floatX)
        y[i] = (target - observed).astype(theano.config.floatX)
    return x, y

if __name__ == "__main__":
    epoch_count = 5
    batches_per_epoch = 500
    batches_per_test_epoch = 100
    batch_size = 64

    # build neural network
    nn = lasagne.layers.InputLayer(shape=(None, 1, ftd.IMG_SIZE))
    input = nn.input_var
    nn = lasagne.layers.DenseLayer(incoming=nn, num_units=40, W=lasagne.init.Normal(0.01), b=lasagne.init.Normal(0.01),
                                   nonlinearity=lasagne.nonlinearities.tanh)
    nn = lasagne.layers.DenseLayer(incoming=nn, num_units=1, W=lasagne.init.Normal(0.01), b=None,
                                   nonlinearity=lasagne.nonlinearities.linear)

    output = lasagne.layers.get_output(nn)
    y = T.matrix('y')
    loss = lasagne.objectives.aggregate(lasagne.objectives.squared_error(y, output))
    params = lasagne.layers.get_all_params(nn)
    updates = lasagne.updates.adam(loss, params, learning_rate=0.001)
    train_fn = theano.function([input, y], loss, updates=updates)
    test_fn = theano.function([input, y], loss)
    forward = theano.function([input], output)

    for e in range(epoch_count):
        # train
        epoch_loss = 0.0
        for b in range(batches_per_epoch):
            progress_bar(b+1, batches_per_epoch, update_freq=batches_per_epoch/100)
            tx, ty = get_batch(batch_size)
            l = train_fn(tx, ty)
            epoch_loss += l
        print "Epoch {0:d}, avg. training loss: {1:f}".format(e+1, epoch_loss/batches_per_epoch)

        # test
        epoch_loss = 0.0
        for b in range(batches_per_test_epoch):
            progress_bar(b+1, batches_per_test_epoch, update_freq=batches_per_test_epoch/100)
            tx, ty = get_batch(batch_size)
            l = test_fn(tx, ty)
            epoch_loss += l
        print "Epoch {0:d}, avg. test loss: {1:f}".format(e+1, epoch_loss/batches_per_test_epoch)

