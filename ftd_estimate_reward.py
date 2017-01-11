import numpy as np

import find_the_dot_problem

import lddp_sampler
import reward


def estimate_reward(chains, seed):
    np.random.seed(seed)
    my_reward = np.zeros(run_count)
    my_accepted = np.zeros(run_count)
    for r in range(run_count):
        print r
        chains.reset()
        for e in range(episodes_per_epoch):
            rs, _, _, accs = chains.run_episode()
            my_reward[r] += np.mean(rs)
            my_accepted[r] += np.mean(accs)
    return my_reward/episodes_per_epoch, my_accepted/episodes_per_epoch


if __name__ == "__main__":
    episode_length = 50
    episodes_per_epoch = 1
    run_count = 50
    my_target = find_the_dot_problem.FindTheDotTargetDistribution(ll_variance=0.2)

    naive_policy = find_the_dot_problem.FindTheDotPolicy(n_hidden=10)
    naive_policy.params[0][:] = 0.0
    naive_policy.params[1][:] = 0.0
    naive_policy.params[2][:] = 0.0
    naive_policy.params[3][:] = 0.0
    naive_policy.params[4][:] = 0.0
    naive_policy.params[5][:] = 0.0

    learned_policy = find_the_dot_problem.FindTheDotPolicy(n_hidden=10)
    p = np.load('results/109115_params.npy')
    for i in range(6):
        learned_policy.params[i] = p[-1][i]

    """
    optimal_policy = find_the_square_problem.FindTheSquarePolicy(sd=np.ones(1))
    optimal_policy.params[0][:, 0] = np.repeat(np.linspace(0.1, -0.1, 20)[0:19], 19)
    optimal_policy.params[0][:, 1] = np.tile(np.linspace(0.1, -0.1, 20)[0:19], 19)
    optimal_policy.params[1][:] = 0.0
    """

    my_chains = lddp_sampler.ParallelMHChains(target_distribution=my_target, policy=naive_policy,
                                              reward_function=reward.log_prob, chain_count=10,
                                              episode_length=episode_length)

    naive_reward, naive_accepted = estimate_reward(my_chains, seed=321)

    my_chains.policy = learned_policy
    learned_reward, learned_accepted = estimate_reward(my_chains, seed=321)

    """
    my_chains.policy = optimal_policy
    optimal_reward, optimal_accepted = estimate_reward(my_chains, seed=123)
    """

