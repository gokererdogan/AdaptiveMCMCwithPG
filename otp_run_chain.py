import sys
import numpy as np

import occluding_tiles_problem as otp

import lddp_sampler
import reward


if __name__ == "__main__":
    np.random.seed(9)

    chain_count = 20
    episode_length = 25
    episode_count = 80
    my_target = otp.OccludingTilesDistribution(ll_sd=0.02)

    my_policy = otp.OccludingTilesPolicy(learn_pick_tile=False, learn_move_tile=True,
                                         move_filter_count=50, move_filter_size=(3, 3),
                                         move_pool_size=(4, 4), move_sd_multiplier=1.0)

    # policy = 'naive'
    # policy = 'supervised2'
    run_id = sys.argv[1]
    print run_id
    policy = 'learned_{0:s}'.format(run_id)
    learned_params = np.load('results/runs/{0:s}_params.npy'.format(run_id))
    # learned_params = np.load('move_tile_supervised_params.npy')
    for i in range(len(my_policy.params)):
        my_policy.params[i] = learned_params[-1][i]
        # my_policy.params[i] = np.reshape(learned_params[i], my_policy.params[i].shape)
        # my_policy.params[i][:] = 0.0

    my_chains = lddp_sampler.ParallelMHChains(target_distribution=my_target, policy=my_policy,
                                              reward_function=reward.log_prob_increase_avg, chain_count=chain_count,
                                              episode_length=episode_length, thinning_period=episode_length)

    """
    rs = np.zeros((episode_count, chain_count))
    for e in range(episode_count):
        print ".",
        sys.stdout.flush()
        results = my_chains.run_episode()
        rs[e] = results[0]
    """

    lls = np.zeros((episode_count, chain_count))
    rmse = np.zeros((episode_count, chain_count))
    ars = np.zeros((episode_count, chain_count))
    for e in range(episode_count):
        results = my_chains.run_episode()
        for c in range(chain_count):
            lls[e, c] = my_target.log_probability(results[2][c][-1])
            rmse[e, c] = np.sqrt(np.mean(np.square((my_target.true_x - results[2][c][-1]) / otp.X_BOUND)))
            ars[e, c] = results[3][c]
        print e, lls[e]
        print e, rmse[e]
        print e, ars[e]

    np.save('lls_{0}.npy'.format(policy), lls)
    np.save('rmse_{0}.npy'.format(policy), rmse)
    np.save('ars_{0}.npy'.format(policy), ars)

    """
    from init_plotting import *
    plt.plot(lls)
    plt.savefig('lls.png')
    plt.figure()
    plt.plot(rmse)
    plt.savefig('rmse.png')
    plt.figure()
    plt.plot(np.mean(lls, axis=1))
    plt.savefig('lls_mean.png')
    plt.figure()
    plt.plot(np.mean(rmse, axis=1))
    plt.savefig('rmse_mean.png')

    # plot the best sample
    c = np.argmax(lls[-1])
    r = otp.OccludingTilesDistribution.render(results[2][c][-1])
    plt.figure()
    plt.imshow(r - my_target.data)
    plt.savefig('diff_{0}.png'.format(policy))
    """
