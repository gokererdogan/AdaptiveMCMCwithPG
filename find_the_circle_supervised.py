"""
Learning data-driven proposals through reinforcement learning

Supervised training of a proposal function for find the circle problem.

22 Sept. 2016
https://github.com/gokererdogan
"""
import theano
import theano.tensor as T
import lasagne

from gmllib.helpers import progress_bar
from find_the_circle_problem import *


def get_batch(size=64):
    x = np.zeros((size, 1) + IMG_SIZE, dtype=theano.config.floatX)
    y = np.zeros((size, 2), dtype=theano.config.floatX)
    for i in range(size):
        target = FindTheCircleProblem()
        observed = FindTheCircleProblem()
        x[i, 0] = (observed.render() - target.render()).astype(theano.config.floatX)
        y[i] = (target.position - observed.position).astype(theano.config.floatX)
    return x, y

if __name__ == "__main__":
    epoch_count = 5
    batches_per_epoch = 500
    batches_per_test_epoch = 100
    batch_size = 64

    # build neural network
    action_dim = 2
    nn = lasagne.layers.InputLayer(shape=(None, 1) + IMG_SIZE)
    input = nn.input_var
    """
    nn = lasagne.layers.Conv2DLayer(incoming=nn, num_filters=2, filter_size=(8, 8), stride=(4, 4),
                                    W=lasagne.init.Normal(0.01), b=lasagne.init.Constant(0.0),
                                    nonlinearity=lasagne.nonlinearities.rectify)
    nn = lasagne.layers.MaxPool2DLayer(incoming=nn, pool_size=(2, 2))
    """
    nn = lasagne.layers.DenseLayer(incoming=nn, num_units=action_dim, W=lasagne.init.Normal(0.01), b=None,
                                   nonlinearity=lasagne.nonlinearities.linear)

    output = lasagne.layers.get_output(nn)
    y = T.matrix('y')
    loss = lasagne.objectives.aggregate(lasagne.objectives.squared_error(y, output))
    params = lasagne.layers.get_all_params(nn)
    updates = lasagne.updates.sgd(loss, params, learning_rate=0.001)
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

