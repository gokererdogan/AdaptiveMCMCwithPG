"""
Learning Data-Driven Proposals Through Reinforcement Learning

This script plots the figures used in our manuscript.

https://github.com/gokererdogan
13 Dec. 2016
"""
import numpy as np
import scipy.stats as stat
from init_plotting import *

import lddp_sampler as sampler
import target_distribution
import policy
import reward


def plot_1D_adjust_sd():
    # read the run
    episodes_per_epoch = 50
    params = np.load('results/1d/214637_params.npy')
    w = np.array([float(e[0][0]) for e in params])
    b = np.array([float(e[1][0]) for e in params])
    plt.figure()
    x = np.arange(0, w.size) * episodes_per_epoch
    plt.plot(x, w)
    plt.plot(x, b)
    plt.hlines(np.log(2.4), x[0], x[-1], linestyles='dashed')
    plt.legend(['w', 'b', 'Optimal b'], loc='best')
    plt.xlabel('Episode')
    plt.ylabel('Parameter values')
    plt.savefig('../latex/fig/1d_adjust_sd.png')

    # calculate autocorrelation time with naive and learned proposals
    def reward_fn(td, xs, accs):
        return reward.auto_correlation_batch_means(td, xs, accs, batch_count=5)

    policy_mean = np.zeros(1)
    policy_sd = None
    my_policy = policy.LinearGaussianPolicy(D=1, mean=policy_mean, sd=policy_sd)

    my_target = target_distribution.MultivariateGaussian(mean=np.zeros(1), cov=np.eye(1))
    my_chains = sampler.ParallelMHChains(target_distribution=my_target, policy=my_policy,
                                         reward_function=reward_fn, chain_count=10,
                                         episode_length=100)
    np.random.seed(3)
    my_chains.reset()
    my_policy.params[0][[0]] = 0.0
    my_policy.params[1][0] = 0.0
    rewards_naive = [np.mean(-my_chains.run_episode()[0]) for _ in range(20)]

    np.random.seed(3)
    my_chains.reset()
    my_policy.params[0][[0]] = w[-1]
    my_policy.params[1][0] = b[-1]
    rewards_learned = [np.mean(-my_chains.run_episode()[0]) for _ in range(20)]

    f = open("../latex/fig/1d_adjust_sd.txt", "w")
    f.write("Autocorrelation with naive: {0:f}+-{1:f}\n".format(np.mean(rewards_naive), stat.sem(rewards_naive)))
    f.write("Autocorrelation with learned: {0:f}+-{1:f}\n".format(np.mean(rewards_learned), stat.sem(rewards_learned)))
    f.close()


def plot_1D_adjust_mean():
    # read the run
    episodes_per_epoch = 50
    params = np.load('results/1d/645309_params.npy')
    w = np.array([float(e[0][0]) for e in params])
    b = np.array([float(e[1][0]) for e in params])
    # plot parameters
    plt.figure()
    x = np.arange(0, w.size) * episodes_per_epoch
    plt.plot(x, w)
    plt.plot(x, b)
    plt.legend(['w', 'b'], loc='best')
    plt.xlabel('Episode')
    plt.ylabel('Parameter values')
    plt.savefig('../latex/fig/1d_adjust_mean.png')

    # plot rewards
    rs = np.load('results/1d/645309_rewards.npy')
    plt.figure()
    x = np.arange(1, rs.size+1)
    plt.plot(x, rs)
    plt.xlabel('Episode')
    plt.ylabel('Negative autocorrelation')
    plt.savefig('../latex/fig/1d_adjust_mean_rewards.png')

    # calculate autocorrelation time with naive and learned proposals
    def reward_fn(td, xs, accs):
        return reward.auto_correlation_batch_means(td, xs, accs, batch_count=5)

    policy_mean = None
    policy_sd = np.ones(1)
    my_policy = policy.LinearGaussianPolicy(D=1, mean=policy_mean, sd=policy_sd)

    my_target = target_distribution.MultivariateGaussian(mean=np.zeros(1), cov=np.eye(1))
    my_chains = sampler.ParallelMHChains(target_distribution=my_target, policy=my_policy,
                                         reward_function=reward_fn, chain_count=10,
                                         episode_length=100)
    np.random.seed(3)
    my_chains.reset()
    my_policy.params[0][[0]] = 0.0
    my_policy.params[1][0] = 0.0
    rewards_naive = [np.mean(-my_chains.run_episode()[0]) for _ in range(20)]

    np.random.seed(3)
    my_chains.reset()
    my_policy.params[0][[0]] = w[-1]
    my_policy.params[1][0] = b[-1]
    rewards_learned = [np.mean(-my_chains.run_episode()[0]) for _ in range(20)]

    f = open("../latex/fig/1d_adjust_mean.txt", "w")
    f.write("Autocorrelation with naive: {0:f}+-{1:f}\n".format(np.mean(rewards_naive), stat.sem(rewards_naive)))
    f.write("Autocorrelation with learned: {0:f}+-{1:f}\n".format(np.mean(rewards_learned), stat.sem(rewards_learned)))
    f.close()

if __name__ == "__main__":
    plot_1D_adjust_sd()
    plot_1D_adjust_mean()
