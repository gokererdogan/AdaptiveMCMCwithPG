"""
Learning Data-Driven Proposals

run_experiment method for running experiments on 1D and 2D Gaussians.

https://github.com/gokererdogan
8 Dec. 2016
"""
import numpy as np
from init_plotting import *

import lddp_sampler as sampler
import gradient_estimator
import target_distribution
import policy
import reward
import parameter_schedule
from optimizer import adam


def run_experiment(**kwargs):
    seed = kwargs['seed']
    results_folder = kwargs['results_folder']
    target_dimensionality = kwargs['target_dimensionality']
    if target_dimensionality == 2:
        target_cov = kwargs['target_cov']
    policy_linearity = kwargs['policy_linearity']
    if policy_linearity == 'nonlinear':
        hidden_count = kwargs['hidden_count']
    policy_type = kwargs['policy_type']
    reward_type = kwargs['reward_type']
    gradient_estimator_type = kwargs['gradient_estimator']
    learning_rate_schedule = kwargs['learning_rate_schedule']
    adam_b1 = kwargs['adam_b1']
    adam_b2 = kwargs['adam_b2']
    gradient_clip = kwargs['gradient_clip']
    chain_count = kwargs['chain_count']
    episode_length = kwargs['episode_length']
    epoch_count = kwargs['epoch_count']
    episodes_per_epoch = kwargs['episodes_per_epoch']
    batch_count = kwargs['batch_count']

    # seed the rng
    np.random.seed(seed)

    if gradient_estimator_type == 'bbvi':
        grad_estimator_fn = gradient_estimator.BBVIEstimator(clip=gradient_clip)
    elif gradient_estimator_type == 'vimco':
        grad_estimator_fn = gradient_estimator.VIMCOEstimator(clip=gradient_clip)
    elif gradient_estimator_type == 'vanilla':
        grad_estimator_fn = gradient_estimator.GradientEstimator(clip=gradient_clip)
    else:
        raise ValueError("Unknown gradient estimator.")

    if reward_type == '-Autocorrelation':
        def reward_fn(td, xs, accs):
            return reward.auto_correlation_batch_means(td, xs, accs, batch_count=batch_count)
    elif reward_type == 'Efficiency':
        def reward_fn(td, xs, accs):
            return reward.efficiency_batch_means(td, xs, accs, batch_count=batch_count)
    elif reward_type == 'Log Probability':
        reward_fn = reward.log_prob
    elif reward_type == 'Acceptance Rate':
        reward_fn = reward.acceptance_rate
    else:
        raise ValueError("Unknown reward type.")

    if policy_type == 'mean':
        policy_mean = None
        policy_sd = np.ones(target_dimensionality)
    elif policy_type == 'sd':
        policy_mean = np.zeros(target_dimensionality)
        policy_sd = None
    elif policy_type == 'mean+sd':
        policy_mean = None
        policy_sd = None
    else:
        raise ValueError("Unknown policy type.")

    if policy_linearity == 'linear':
        my_policy = policy.LinearGaussianPolicy(D=target_dimensionality, mean=policy_mean, sd=policy_sd)
    elif policy_linearity == 'nonlinear':
        my_policy = policy.NonlinearGaussianPolicy(D=target_dimensionality, n_hidden=hidden_count, mean=policy_mean,
                                                 sd=policy_sd)
    else:
        raise ValueError("Unknown policy linearity.")

    if target_dimensionality == 1:
        my_target = target_distribution.MultivariateGaussian(mean=np.zeros(1), cov=np.eye(1))
    elif target_dimensionality == 2:
        my_target = target_distribution.MultivariateGaussian(mean=np.zeros(2), cov=target_cov)
    else:
        raise ValueError("Target dimensionality can be 1 or 2.")

    my_chains = sampler.ParallelMHChains(target_distribution=my_target, policy=my_policy,
                                         reward_function=reward_fn, chain_count=chain_count,
                                         episode_length=episode_length)

    params, rewards, grad_magnitudes = adam(my_chains, grad_estimator_fn,
                                            learning_rate_schedule=learning_rate_schedule,
                                            epoch_count=epoch_count, episodes_per_epoch=episodes_per_epoch,
                                            report_period=1, b1=adam_b1, b2=adam_b2)

    # generate a random run id
    np.random.seed()
    run_id = np.random.randint(1000000)
    results_file_prefix = "{0:s}/{1:06d}".format(results_folder, run_id)

    # save results
    params_file = "{0:s}_params.npy".format(results_file_prefix)
    np.save(params_file, params)
    rewards_file = "{0:s}_rewards.npy".format(results_file_prefix)
    np.save(rewards_file, rewards)
    grad_magnitudes_file = "{0:s}_grad_magnitudes.npy".format(results_file_prefix)
    np.save(grad_magnitudes_file, grad_magnitudes)

    # plot rewards and grad_magnitudes
    plt.figure()
    plt.plot(rewards)
    plt.xlabel('Iteration')
    plt.ylabel(reward_type)
    rewards_plot_file = "{0:s}_rewards.png".format(results_file_prefix)
    plt.savefig(rewards_plot_file)

    for i in range(grad_magnitudes.shape[1]):
        plt.figure()
        plt.plot(grad_magnitudes[:, i])
        plt.xlabel('Iteration')
        plt.ylabel('Param {0:d} gradient magnitude'.format(i))
        plot_file = "{0:s}_grad_magnitudes_{1:d}.png".format(results_file_prefix, i)
        plt.savefig(plot_file)

    # form the results dictionary
    results = {'run_id': run_id}
    return results

