"""
Learning Data-Driven Proposals

run_experiment method for running experiments.

https://github.com/gokererdogan
8 Dec. 2016
"""
import numpy as np
from init_plotting import *

import lddp_sampler as sampler
from optimizer import adam


def run_experiment(**kwargs):
    seed = kwargs['seed']
    results_folder = kwargs['results_folder']
    my_target = kwargs['target_distribution']
    my_policy = kwargs['policy']
    reward_fn = kwargs['reward_function']
    reward_type = kwargs['reward_type']
    my_grad_estimator = kwargs['gradient_estimator']
    learning_rate_schedule = kwargs['learning_rate_schedule']
    adam_b1 = kwargs['adam_b1']
    adam_b2 = kwargs['adam_b2']
    chain_count = kwargs['chain_count']
    episode_length = kwargs['episode_length']
    epoch_count = kwargs['epoch_count']
    episodes_per_epoch = kwargs['episodes_per_epoch']
    save_period = kwargs['save_period']
    plot_period = kwargs['plot_period']
    report_period = kwargs['report_period']
    load_from_run = kwargs['load_from_run']

    # generate a random run id
    np.random.seed()
    run_id = np.random.randint(1000000)
    results_file_prefix = "{0:s}/{1:06d}".format(results_folder, run_id)

    # seed the rng
    np.random.seed(seed)
    my_target.reset()

    # load params from an earlier run
    if load_from_run is not None:
        load_params = np.load('{0:s}/{1:06d}_params.npy'.format(results_folder, load_from_run))
        for i, pi in enumerate(load_params[-1]):
            my_policy.params[i][:] = pi

    my_chains = sampler.ParallelMHChains(target_distribution=my_target, policy=my_policy,
                                         reward_function=reward_fn, chain_count=chain_count,
                                         episode_length=episode_length)

    # callback functions for reporting and plotting
    def report_callback(iteration, rs, ars):
        print "Iteration {0:d}\tAvg. Reward: {1:.3f}, " \
              "Acceptance Rate: {2:.3f}".format(iteration+1, np.mean(rs[0:(iteration+1)]),
                                                np.mean(ars[0:(iteration+1)]))

    def plot_callback(iteration, rs, ars):
        # plot rewards and grad_magnitudes
        plt.figure()
        plt.plot(rs[0:(iteration+1)])
        plt.xlabel('Iteration')
        plt.ylabel(reward_type)
        rewards_plot_file = "{0:s}_rewards.png".format(results_file_prefix)
        plt.savefig(rewards_plot_file)

        plt.figure()
        plt.plot(ars[0:(iteration+1)])
        plt.xlabel('Iteration')
        plt.ylabel('Acceptance rate')
        ar_plot_file = "{0:s}_acceptance_rate.png".format(results_file_prefix)
        plt.savefig(ar_plot_file)
        plt.close()

    params, rewards, grad_magnitudes, acceptance_rates = adam(my_chains, my_grad_estimator,
                                                              learning_rate_schedule=learning_rate_schedule,
                                                              epoch_count=epoch_count,
                                                              episodes_per_epoch=episodes_per_epoch,
                                                              report_period=report_period,
                                                              save_period=save_period,
                                                              plot_period=plot_period,
                                                              report_callback=report_callback,
                                                              plot_callback=plot_callback,
                                                              b1=adam_b1, b2=adam_b2)

    # save results
    params_file = "{0:s}_params.npy".format(results_file_prefix)
    np.save(params_file, params)
    rewards_file = "{0:s}_rewards.npy".format(results_file_prefix)
    np.save(rewards_file, rewards)
    grad_magnitudes_file = "{0:s}_grad_magnitudes.npy".format(results_file_prefix)
    np.save(grad_magnitudes_file, grad_magnitudes)
    acceptance_rates_file = "{0:s}_acceptance_rates.npy".format(results_file_prefix)
    np.save(acceptance_rates_file, acceptance_rates)

    # plot rewards and acceptance rates
    plot_callback(len(rewards), rewards, acceptance_rates)

    for i in range(grad_magnitudes.shape[1]):
        plt.figure()
        plt.plot(grad_magnitudes[:, i])
        plt.xlabel('Iteration')
        plt.ylabel('Param {0:d} gradient magnitude'.format(i))
        plot_file = "{0:s}_grad_magnitudes_{1:d}.png".format(results_file_prefix, i)
        plt.savefig(plot_file)

    # form the results dictionary
    results = {'run_id': run_id, 'avg_reward': np.mean(rewards)}
    return results

