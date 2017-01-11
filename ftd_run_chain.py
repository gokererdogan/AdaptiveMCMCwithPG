import numpy as np

import find_the_dot_problem as ftd

import lddp_sampler
import reward


if __name__ == "__main__":
    np.random.seed(1)

    episode_length = 50
    my_target = ftd.FindTheDotTargetDistribution(ll_variance=0.2)

    my_policy = ftd.FindTheDotPolicy(n_hidden=10)
    # p = np.load('results/194494_params.npy')
    for i in range(6):
        # my_policy.params[i] = p[-1][i]
        my_policy.params[i][:] = 0.0

    my_chains = lddp_sampler.ParallelMHChains(target_distribution=my_target, policy=my_policy,
                                              reward_function=reward.log_prob, chain_count=1,
                                              episode_length=episode_length, thinning_period=1)

    # my_chains.x0 = np.array([[3.0]])
    results = my_chains.run_episode()
    print results[2]
    print results[3]
