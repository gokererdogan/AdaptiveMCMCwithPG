"""
Learning data-driven proposals through reinforcement learning

Supervised training of a proposal function for occluding tiles problem.

27 Sept. 2016
https://github.com/gokererdogan
"""
import theano
import theano.tensor as T
import lasagne

from gmllib.helpers import progress_bar
from occluding_tiles_problem import *


def get_batch(size=64):
    actions = [move_tile, resize_tile, rotate_tile]
    x = np.zeros((size, 1) + IMG_SIZE, dtype=theano.config.floatX)
    y = np.zeros((size, len(actions)), dtype=theano.config.floatX)
    for i in range(size):
        target = OccludingTilesHypothesis(tile_count=1)
        action_ix = np.random.randint(3)
        observed, _, _ = actions[action_ix](target, None)
        x[i, 0] = (observed.render() - target.render()).astype(theano.config.floatX)
        y[i, action_ix] = 1.0
    return x, y

if __name__ == "__main__":
    epoch_count = 10
    batches_per_epoch = 500
    batches_per_test_epoch = 100
    batch_size = 64

    # build neural network
    action_dim = 3
    nn = lasagne.layers.InputLayer(shape=(None, 1) + IMG_SIZE)
    input = nn.input_var
    """
    nn = lasagne.layers.Conv2DLayer(incoming=nn, num_filters=32, filter_size=(3, 3), stride=(1, 1),
                                    nonlinearity=lasagne.nonlinearities.rectify)
    nn = lasagne.layers.MaxPool2DLayer(incoming=nn, pool_size=(2, 2))
    nn = lasagne.layers.Conv2DLayer(incoming=nn, num_filters=8, filter_size=(6, 6), stride=(2, 2),
                                    nonlinearity=lasagne.nonlinearities.rectify)
    nn = lasagne.layers.MaxPool2DLayer(incoming=nn, pool_size=(2, 2))
    """
    nn = lasagne.layers.DenseLayer(incoming=nn, num_units=500, nonlinearity=lasagne.nonlinearities.rectify)
    nn = lasagne.layers.DenseLayer(incoming=nn, num_units=action_dim, W=lasagne.init.Normal(0.01), b=None,
                                   nonlinearity=lasagne.nonlinearities.softmax)

    output = lasagne.layers.get_output(nn)
    y = T.matrix('y')
    loss = lasagne.objectives.aggregate(lasagne.objectives.categorical_crossentropy(output, y))
    accuracy = T.mean(T.eq(T.argmax(output, axis=1), T.argmax(y, axis=1)))
    params = lasagne.layers.get_all_params(nn)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01)
    train_fn = theano.function([input, y], loss, updates=updates)
    test_fn = theano.function([input, y], accuracy)
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
        epoch_acc = 0.0
        for b in range(batches_per_test_epoch):
            progress_bar(b+1, batches_per_test_epoch, update_freq=batches_per_test_epoch/100)
            tx, ty = get_batch(batch_size)
            a = test_fn(tx, ty)
            epoch_acc += a
        print "Epoch {0:d}, avg. test accuracy: {1:f}".format(e+1, epoch_acc/batches_per_test_epoch)

    # test nn
    tx, ty = get_batch(3)
    py = forward(tx)
    print "Labels"
    print ty
    print "Predictions"
    print py


