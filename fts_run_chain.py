import numpy as np

import find_the_square_problem as fts

import lddp_sampler
import reward


if __name__ == "__main__":
    # np.random.seed(1)

    episode_length = 1000
    my_target = fts.FindTheSquareTargetDistribution(ll_variance=5.0)

    my_policy = fts.FindTheSquarePolicy(n_hidden=50)
    for p in my_policy.params:
        p[:] = 0.0

    my_chains = lddp_sampler.ParallelMHChains(target_distribution=my_target, policy=my_policy,
                                              reward_function=reward.log_prob, chain_count=1,
                                              episode_length=episode_length, thinning_period=10)

    # my_chains.x0 = np.array([[0.0, 0.0]])
    results = my_chains.run_episode()
    print results[2]
    print results[3]
