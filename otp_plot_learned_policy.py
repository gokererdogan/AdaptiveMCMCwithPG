import numpy as np
import matplotlib.pyplot as plt

import occluding_tiles_problem as otp


def plot_sample(close_to_true=True):
    true_x = my_target.initial_x()
    data = my_target.render(true_x)
    data_rgb = my_target.render_rgb(true_x)

    if close_to_true:
        current_x = true_x.copy()
        i = np.random.randint(6)
        current_x[i*4:(i+1)*4] += np.random.randn(4) * 2.0
    else:
        current_x = my_target.initial_x()

    img = my_target.render(current_x)
    img_rgb = my_target.render_rgb(current_x)

    tile_probs = my_policy.get_pick_tile_proposal(current_x, data, my_policy.params)
    print "\n".join(["{0}: {1}".format(c, p) for c, p in zip(otp.COLORS, tile_probs)])
    print
    tile_ix = np.argmax(tile_probs)
    m, sd = my_policy.get_move_tile_proposal(current_x, data, tile_ix, my_policy.params)
    print "Picked tile color: {0}".format(otp.COLORS[tile_ix])
    print "Picked tile params: {0}".format(current_x[tile_ix*4:(tile_ix+1)*4])
    print "True tile params: {0}".format(true_x[tile_ix*4:(tile_ix+1)*4])
    print "Move: {0}-{1}".format(m, sd)

    if close_to_true and tile_ix != i:
        print
        m, sd = my_policy.get_move_tile_proposal(current_x, data, i, my_policy.params)
        print "Moved tile color: {0}".format(otp.COLORS[i])
        print "Moved tile params: {0}".format(current_x[i*4:(i+1)*4])
        print "True tile params: {0}".format(true_x[i*4:(i+1)*4])
        print "Move: {0}-{1}".format(m, sd)

    plt.subplot(321)
    plt.imshow(data_rgb)
    plt.ylabel('data')
    plt.subplot(322)
    plt.imshow(img_rgb)
    plt.ylabel('img')
    plt.subplot(323)
    plt.imshow(img_rgb - data_rgb)
    plt.ylabel('diff')
    plt.subplot(324)
    plt.imshow(np.abs(img_rgb - data_rgb))
    plt.ylabel('abs diff')

    plt.subplot(325)
    plt.imshow(img[:, :, tile_ix] - data[:, :, tile_ix])
    plt.ylabel('picked tile')

    if close_to_true:
        plt.subplot(326)
        plt.imshow(img[:, :, i] - data[:, :, i])
        plt.ylabel('moved tile')

    plt.show()


if __name__ == "__main__":
    run_no = 641333
    params = np.load('results/{0:06}_params.npy'.format(run_no))
    rewards = np.load('results/{0:06}_rewards.npy'.format(run_no))

    learn_pick = False

    if learn_pick:
        my_policy = otp.OccludingTilesPolicy(learn_pick_tile=True, learn_move_tile=True,
                                             pick_filter_count=20, move_filter_count=50,
                                             move_filter_size=(3, 3), move_pool_size=(4, 4),
                                             move_sd_multiplier=1.0)
    else:
        my_policy = otp.OccludingTilesPolicy(learn_pick_tile=False, learn_move_tile=True,
                                             move_filter_count=50, move_filter_size=(3, 3),
                                             move_pool_size=(4, 4), move_sd_multiplier=1.0)

    """
    learned_params = np.load('move_tile_supervised_params.npy')
    for i in range(len(my_policy.params)):
        # my_policy.params[i] = learned_params[-1][i]
        my_policy.params[i] = np.reshape(learned_params[i], my_policy.params[i].shape)
        # my_policy.params[i][:] = 0.0
    """

    my_policy.params = params[-1]

    my_target = otp.OccludingTilesDistribution(ll_sd=0.02)
    plot_sample(close_to_true=True)

