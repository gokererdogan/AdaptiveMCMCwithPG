"""
Learning data-driven proposals through reinforcement learning

Supervised training of a pick tile proposal function for occluding tiles problem.

13 Jan. 2017
https://github.com/gokererdogan
"""
import numpy as np

import theano
import theano.tensor as T
import lasagne

from gmllib.helpers import progress_bar
import occluding_tiles_problem as otp

target_distribution = otp.OccludingTilesDistribution(ll_sd=1.0)
img_mean = np.load('otp_mean.npy')


def get_batch(size=64):
    x = np.zeros((size, 2, 3, otp.IMG_SIZE[0], otp.IMG_SIZE[1]), dtype=theano.config.floatX)
    y = np.zeros((size, 6), dtype=theano.config.floatX)
    for i in range(size):
        target = target_distribution.initial_x()
        observed = target.copy()
        tile_ix = np.random.randint(6)
        observed[tile_ix*4:(tile_ix+1)*4] += np.random.randn(4) * 2.0
        x[i, 0] = (np.transpose(otp.OccludingTilesDistribution.render(observed), (2, 0, 1))).astype(theano.config.floatX) - img_mean
        x[i, 1] = (np.transpose(otp.OccludingTilesDistribution.render(target), (2, 0, 1))).astype(theano.config.floatX) - img_mean
        y[i, tile_ix] = 1.0
    return x, y


def get_batch_6D(size=64):
    x = np.zeros((size, 6, otp.IMG_SIZE[0], otp.IMG_SIZE[1]), dtype=theano.config.floatX)
    y = np.zeros((size, 6), dtype=theano.config.floatX)
    for i in range(size):
        target = target_distribution.initial_x()
        timg = otp.OccludingTilesDistribution.render(target)
        t6d = np.zeros((6, 50, 50))
        for ci, c in enumerate(otp.COLORS):
            t6d[ci, np.all(np.abs(timg - c) < 1e-3, axis=2)] = 1.0
        observed = target.copy()
        tile_ix = np.random.randint(6)
        observed[tile_ix*4:(tile_ix+1)*4] += np.random.randn(4) * 2.0
        oimg = otp.OccludingTilesDistribution.render(observed)
        o6d = np.zeros((6, 50, 50))
        for ci, c in enumerate(otp.COLORS):
            o6d[ci, np.all(np.abs(oimg - c) < 1e-3, axis=2)] = 1.0
        x[i] = (o6d - t6d).astype(theano.config.floatX)
        y[i, tile_ix] = 1.0
    return x, y


def get_batch_masked(size=64):
    x = np.zeros((size, 12, otp.IMG_SIZE[0], otp.IMG_SIZE[1]), dtype=theano.config.floatX)
    y = np.zeros((size, 6), dtype=theano.config.floatX)
    for i in range(size):
        target = target_distribution.initial_x()
        observed = target.copy()
        tile_ix = np.random.randint(6)
        observed[tile_ix*4:(tile_ix+1)*4] += np.random.randn(4) * 2.0
        oimg = otp.OccludingTilesDistribution.render(observed)
        timg = otp.OccludingTilesDistribution.render(target)
        for ci, c in enumerate(otp.COLORS):
            mobs = np.zeros(otp.IMG_SIZE)
            mobs[np.all(np.abs(oimg - c) < 1e-3, axis=2)] = 1.0
            mtar = np.zeros(otp.IMG_SIZE)
            mtar[np.all(np.abs(timg - c) < 1e-3, axis=2)] = 1.0
            x[i, ci] = mobs
            x[i, ci+6] = mtar
        y[i, tile_ix] = 1.0
    return x, y


def classify_pixel_mass(x, y):
    # classify based on pixel mass
    # intuitively, this should be the mechanism learned by neural network
    # this is rather successful with accuracy around 0.97
    correct = 0
    for xi, yi in zip(x, y):
        obs = np.transpose(xi[0] + img_mean, (1, 2, 0))
        tar = np.transpose(xi[1] + img_mean, (1, 2, 0))
        color_mass = np.zeros(6)
        for ci, c in enumerate(otp.COLORS):
            mobs = np.zeros(otp.IMG_SIZE)
            mobs[np.all(np.abs(obs - c) < 10e-3, axis=2)] = 1.0
            mtar = np.zeros(otp.IMG_SIZE)
            mtar[np.all(np.abs(tar - c) < 10e-3, axis=2)] = 1.0
            color_mass[ci] = np.sum(np.square(mobs - mtar))
        pred_y = np.argmax(color_mass)
        if np.argmax(yi) == pred_y:
            correct += 1
    return correct / float(x.shape[0])


def train(epoch_count, batch_size, batches_per_epoch, batches_per_test_epoch, type='original'):
    for e in range(epoch_count):
        # train
        epoch_loss = 0.0
        for b in range(batches_per_epoch):
            progress_bar(b+1, batches_per_epoch, update_freq=batches_per_epoch/100 or 1)
            if type == 'masked':
                tx, ty = get_batch_masked(batch_size)
                l = train_fn(tx, ty)
            elif type == '6d':
                tx, ty = get_batch_6D(batch_size)
                l = train_fn(tx, ty)
            else:
                tx, ty = get_batch(batch_size)
                l = train_fn(tx[:, 0], tx[:, 1], ty)
            if np.isnan(l):
                raise RuntimeError("Nan!!!")
            epoch_loss += l
        epoch_loss /= batches_per_epoch
        print "Epoch {0:d}, avg. training loss: {1:f}".format(e+1, epoch_loss)

        # test
        epoch_loss = 0.0
        epoch_acc = 0.0
        for b in range(batches_per_test_epoch):
            progress_bar(b+1, batches_per_test_epoch, update_freq=batches_per_test_epoch/100 or 1)
            if type == 'masked':
                tx, ty = get_batch_masked(batch_size)
                l = test_fn(tx, ty)
                a = accuracy_fn(tx, ty)
            elif type == '6d':
                tx, ty = get_batch_6D(batch_size)
                l = test_fn(tx, ty)
                a = accuracy_fn(tx, ty)
            else:
                tx, ty = get_batch(batch_size)
                l = test_fn(tx[:, 0], tx[:, 1], ty)
                a = accuracy_fn(tx[:, 0], tx[:, 1], ty)
            epoch_loss += l
            epoch_acc += a
        epoch_loss /= batches_per_test_epoch
        epoch_acc /= batches_per_test_epoch
        print "Epoch {0:d}, avg. test loss: {1:f}, " \
              "avg. test acc: {2:f}".format(e+1, epoch_loss, epoch_acc)


if __name__ == "__main__":
    epoch_count = 5
    batches_per_epoch = 50
    batches_per_test_epoch = 20
    batch_size = 64

    # build neural network
    """
    in1 = lasagne.layers.InputLayer(shape=(None, 3, otp.IMG_SIZE[0], otp.IMG_SIZE[1]))
    in2 = lasagne.layers.InputLayer(shape=(None, 3, otp.IMG_SIZE[0], otp.IMG_SIZE[1]))
    input1 = in1.input_var
    input2 = in2.input_var

    conv1 = lasagne.layers.Conv2DLayer(incoming=in1, num_filters=20, filter_size=(1, 1), stride=1,
                                       W=lasagne.init.Normal(0.01), b=lasagne.init.Normal(0.01),
                                       nonlinearity=lasagne.nonlinearities.rectify)
    conv2 = lasagne.layers.Conv2DLayer(incoming=in2, num_filters=20, filter_size=(1, 1), stride=1,
                                       W=lasagne.init.Normal(0.01), b=lasagne.init.Normal(0.01),
                                       # W=conv1.W, b=conv1.b,
                                       nonlinearity=lasagne.nonlinearities.rectify)
    nn = lasagne.layers.ConcatLayer(incomings=[conv1, conv2], axis=1)
    """
    nn = lasagne.layers.InputLayer(shape=(None, 6, 50, 50))
    input = nn.input_var
    nn = lasagne.layers.Conv2DLayer(incoming=nn, num_filters=20, filter_size=(2, 2), stride=2,
                                    W=lasagne.init.Normal(0.001), b=lasagne.init.Normal(0.001),
                                    nonlinearity=lasagne.nonlinearities.rectify)
    nn = lasagne.layers.GlobalPoolLayer(incoming=nn)
    """
    nn = lasagne.layers.Conv2DLayer(incoming=nn, num_filters=10, filter_size=(1, 1), stride=1,
                                    W=lasagne.init.Normal(0.01), b=lasagne.init.Normal(0.01),
                                    nonlinearity=lasagne.nonlinearities.tanh)
    nn = lasagne.layers.DenseLayer(incoming=nn, num_units=50, W=lasagne.init.Normal(0.01), b=lasagne.init.Normal(0.01),
                                   nonlinearity=lasagne.nonlinearities.rectify)
                                   """
    nn = lasagne.layers.DenseLayer(incoming=nn, num_units=6,
                                   W=lasagne.init.Normal(0.001), b=lasagne.init.Normal(0.001),
                                   nonlinearity=lasagne.nonlinearities.softmax)

    output = lasagne.layers.get_output(nn)
    y = T.matrix('y')
    loss = lasagne.objectives.aggregate(lasagne.objectives.categorical_crossentropy(output, y))
    accuracy = T.mean(T.eq(T.argmax(y, axis=1), T.argmax(output, axis=1)))
    params = lasagne.layers.get_all_params(nn)
    updates = lasagne.updates.adam(loss, params, learning_rate=0.01)
    """
    train_fn = theano.function([input1, input2, y], loss, updates=updates)
    test_fn = theano.function([input1, input2, y], loss)
    accuracy_fn = theano.function([input1, input2, y], accuracy)
    forward = theano.function([input1, input2], output)
    """
    train_fn = theano.function([input, y], loss, updates=updates)
    test_fn = theano.function([input, y], loss)
    accuracy_fn = theano.function([input, y], accuracy)
    forward = theano.function([input], output)

    # train()

