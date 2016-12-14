"""
Learning Data-Driven Proposals

Plots the reward (e.g., auto-correlation time, efficiency, acceptance rate) surface
with respect to parameters of the data-driven proposal for various settings.

https://github.com/gokererdogan
5 Dec. 2016
"""
import autograd.numpy as np
import reward
import target_distribution
import policy
import lddp_sampler as sampler


def estimate_reward_surface_wb(chains, seed=None):
    if seed is None:
        seed = np.random.randint(2**32 - 1, dtype=np.uint32)
    wr = np.linspace(-3, 3, 25)
    br = np.linspace(-2, 2, 25)
    rewards = np.zeros((25, 25))
    for i, w in enumerate(wr):
        chains.policy.params[0][0][0] = w
        for j, b in enumerate(br):
            chains.policy.params[1][0] = b
            np.random.seed(seed)
            chains.reset()
            rs, _ = chains.run_episode()
            rewards[i, j] = np.mean(rs)
            print i*25 + j, wr[i], br[j], rewards[i, j]

    return wr, br, rewards


def estimate_reward_surface_b(chains, seed=None):
    if seed is None:
        seed = np.random.randint(2**32 - 1, dtype=np.uint32)
    # vr = np.linspace(0.01, 10.0, 100)
    # br = np.log(vr)
    br = np.linspace(-2, 2, 100)
    rewards = np.zeros(100)
    chains.policy.params[0][0][0] = 0.0
    for j, b in enumerate(br):
        chains.policy.params[1][0] = b
        np.random.seed(seed)
        chains.reset()
        rs, _ = chains.run_episode()
        rewards[j] = np.mean(rs)
        print j, br[j], rewards[j]

    return br, rewards


def estimate_reward_surface_w(chains, seed=None):
    if seed is None:
        seed = np.random.randint(2**32 - 1, dtype=np.uint32)
    wr = np.linspace(-3, 3, 100)
    rewards = np.zeros(100)
    chains.policy.params[1][0] = 0.0
    for j, w in enumerate(wr):
        chains.policy.params[0][0][0] = w
        np.random.seed(seed)
        chains.reset()
        rs, _ = chains.run_episode()
        rewards[j] = np.mean(rs)
        print j, wr[j], rewards[j]

    return wr, rewards


if __name__ == "__main__":
    # use batch means estimate
    def reward_batch_means(td, xs, ac):
        return reward.auto_correlation_batch_means(td, xs, ac, batch_count=8)

    """
    
    # plot autocorrelation time with respect to variance
    my_target = lddp.MultivariateGaussian(mean=np.zeros(1), cov=np.ones(1))
    my_policy = lddp.GaussianPolicy(D=1, mean=np.zeros(1))

    my_chains = lddp.ParallelMHChains(target_distribution=my_target, policy=my_policy,
                                      reward_function=reward_batch_means,
                                      chain_count=10, episode_length=496)

    # # adjust parameter b
    br, rs = estimate_reward_surface_b(my_chains, seed=123)
    np.save('autocorrelation_time_wrt_variance_b.npy', [br, rs])

    # # adjust parameter w
    wr, rs = estimate_reward_surface_w(my_chains, seed=123)
    np.save('autocorrelation_time_wrt_variance_w.npy', [wr, rs])

    # # adjust both
    wr, br, rs = estimate_reward_surface_wb(my_chains, seed=123)
    np.save('autocorrelation_time_wrt_variance_wb.npy', [wr, br, rs])
    """
    # plot autocorrelation time with respect to mean
    my_target = target_distribution.MultivariateGaussian(mean=np.zeros(1), cov=np.ones(1))
    my_policy = policy.LinearGaussianPolicy(D=1, sd=np.ones(1))

    my_chains = sampler.ParallelMHChains(target_distribution=my_target, policy=my_policy,
                                         reward_function=reward_batch_means,
                                         chain_count=10, episode_length=496)

    """
    # # adjust parameter b
    br, rs = estimate_reward_surface_b(my_chains, seed=123)
    np.save('autocorrelation_time_wrt_mean_b.npy', [br, rs])

    # # adjust parameter w
    wr, rs = estimate_reward_surface_w(my_chains, seed=123)
    np.save('autocorrelation_time_wrt_mean_w.npy', [wr, rs])
    """

    # # adjust both
    wr, br, rs = estimate_reward_surface_wb(my_chains, seed=1)
    np.save('autocorrelation_time_wrt_mean_wb.npy', rs)

