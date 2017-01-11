import numpy as np

import occluding_tiles_problem as otp

import lddp_sampler
import reward


if __name__ == "__main__":
    # np.random.seed(2)

    chain_count = 1
    episode_length = 1000
    episode_count = 1
    my_target = otp.OccludingTilesDistribution(ll_sd=0.02)

    my_policy = otp.OccludingTilesPolicy(sd_multiplier=1.0, n_hidden=50)
    for p in my_policy.params:
        p[:] = 0.0

    my_chains = lddp_sampler.ParallelMHChains(target_distribution=my_target, policy=my_policy,
                                              reward_function=reward.log_prob_increase_avg, chain_count=chain_count,
                                              episode_length=episode_length, thinning_period=10)

    results = my_chains.run_episode()
    print results[2]
    print results[3]

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
    plt.savefig('diff.png')
    """
