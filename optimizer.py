"""
Learning data-driven proposals through reinforcement learning

This file contains the implementation of Adam (Kingma & Ba, 2014) optimizer.

13 Dec. 2016
https://github.com/gokererdogan
"""
from gmllib.helpers import progress_bar

import numpy as np


def sga(chains, gradient_estimator, learning_rate_schedule, epoch_count=100,
        episodes_per_epoch=10, save_period=1, report_period=10, plot_period=10,
        report_callback=None, plot_callback=None):
    """
    Stochastic Gradient ascent.
    """
    # we don't want to skip saving the last state of parameters, so make sure that
    # the number of iterations is a multiple of save period.
    if (epoch_count * episodes_per_epoch) % save_period != 0:
        raise ValueError("Number of iterations needs to be a multiple of save period.")

    param_count = len(chains.policy.params)
    max_iteration = epoch_count * episodes_per_epoch
    rewards = np.zeros(max_iteration)
    acceptance_rates = np.zeros(max_iteration)
    grad_magnitudes = np.zeros((max_iteration, param_count))
    params = list()

    # store initial params
    ps = []
    for p in chains.policy.params:
        ps.append(p.copy())
    params.append(ps)

    for epoch in range(epoch_count):
        for episode in range(episodes_per_epoch):
            iteration = (epoch * episodes_per_epoch) + episode
            # progress_bar(iteration+1, max_iteration, update_freq=max_iteration/100 or 1)
            rs, dps, xs, ars = chains.run_episode()
            rewards[iteration] = np.mean(rs)
            acceptance_rates[iteration] = np.mean(ars)

            g = gradient_estimator.estimate_gradient(rs, dps)
            learning_rate = learning_rate_schedule.get_value(iteration)
            for i, gi in enumerate(g):
                grad_magnitudes[iteration, i] = np.sum(np.square(gi))
                chains.policy.params[i] += learning_rate * gi

            # call report and plot callback functions
            if (iteration+1) % report_period == 0 and report_callback is not None:
                report_callback(iteration, rewards, acceptance_rates)

            if (iteration+1) % plot_period == 0 and plot_callback is not None:
                plot_callback(iteration, rewards, acceptance_rates)

            # store params
            if (iteration+1) % save_period == 0:
                ps = []
                for p in chains.policy.params:
                    ps.append(p.copy())
                params.append(ps)

        # reset the chains (i.e., start from random states)
        chains.reset()

    return params, rewards, grad_magnitudes, acceptance_rates


def adam(chains, gradient_estimator, learning_rate_schedule, epoch_count=100,
         episodes_per_epoch=10, save_period=1, report_period=10, plot_period=10,
         b1=0.9, b2=0.999, eps=10**-8,
         report_callback=None, plot_callback=None):
    """
    Gradient ascent with adam.
    Taken from autograd's adam implementation
    """
    # we don't want to skip saving the last state of parameters, so make sure that
    # the number of iterations is a multiple of save period.
    if (epoch_count * episodes_per_epoch) % save_period != 0:
        raise ValueError("Number of iterations needs to be a multiple of save period.")

    param_count = len(chains.policy.params)
    max_iteration = epoch_count * episodes_per_epoch
    rewards = np.zeros(max_iteration)
    acceptance_rates = np.zeros(max_iteration)
    grad_magnitudes = np.zeros((max_iteration, param_count))
    params = list()
    m = [np.zeros_like(p) for p in chains.policy.params]
    v = [np.zeros_like(p) for p in chains.policy.params]

    # store initial params
    ps = []
    for p in chains.policy.params:
        ps.append(p.copy())
    params.append(ps)

    for epoch in range(epoch_count):
        for episode in range(episodes_per_epoch):
            iteration = (epoch * episodes_per_epoch) + episode
            # progress_bar(iteration+1, max_iteration, update_freq=max_iteration/100 or 1)
            rs, dps, xs, ars = chains.run_episode()
            rewards[iteration] = np.mean(rs)
            acceptance_rates[iteration] = np.mean(ars)

            learning_rate = learning_rate_schedule.get_value(iteration)
            g = gradient_estimator.estimate_gradient(rs, dps)
            for i, gi in enumerate(g):
                m[i] = (1 - b1) * gi + b1 * m[i]  # First  moment estimate.
                v[i] = (1 - b2) * (gi**2) + b2 * v[i]  # Second moment estimate.
                mhat = m[i] / (1 - b1**(iteration + 1))    # Bias correction.
                vhat = v[i] / (1 - b2**(iteration + 1))
                chains.policy.params[i] += learning_rate * (mhat / (np.sqrt(vhat) + eps))
                grad_magnitudes[iteration, i] = np.sum(np.square(mhat / (np.sqrt(vhat) + eps)))

            # call report and plot callback functions
            if (iteration+1) % report_period == 0 and report_callback is not None:
                report_callback(iteration, rewards, acceptance_rates)

            if (iteration+1) % plot_period == 0 and plot_callback is not None:
                plot_callback(iteration, rewards, acceptance_rates)

            # store params
            if (iteration+1) % save_period == 0:
                ps = []
                for p in chains.policy.params:
                    ps.append(p.copy())
                params.append(ps)

        # reset the chains (i.e., start from random states)
        chains.reset()

    return params, rewards, grad_magnitudes, acceptance_rates

