"""
Learning data-driven proposals through reinforcement learning

This file contains the implementations of various reward functions for measuring the goodness of a
transition kernel.

13 Dec. 2016
https://github.com/gokererdogan
"""
import autograd.numpy as np


def log_prob(target_distribution, x0, xs, accepteds):
    """
    Returns average log probability of samples xs.
    """
    return np.mean([target_distribution.log_probability(x) for x in xs])


def log_prob_increase(target_distribution, x0, xs, accepteds):
    """
    Returns increase in log probability of samples xs.
    """
    return target_distribution.log_probability(xs[-1]) - target_distribution.log_probability(x0)


def log_prob_increase_avg(target_distribution, x0, xs, accepteds):
    return np.mean([target_distribution.log_probability(x) for x in xs]) - target_distribution.log_probability(x0)


def acceptance_rate(target_distribution, x0, xs, accepteds):
    """
    Returns acceptance rate for samples xs.
    """
    return np.mean(accepteds)


def auto_correlation_naive(target_distribution, x0, xs, accepteds, max_lag=None):
    """
    Calculates negative auto correlation of samples xs.
    This uses the usual window estimators for auto correlation and sums these to get an estimate
    of autocorrelation time (i.e., Monte Carlo error). However, note this is not even a consistent
    estimator of autocorrelation time. (See Geyer (1992))

    Because we use this as a reward function and higher auto correlations
    are worse, we return the negative of auto correlation.

    Parameters:
        xs (np.ndarray): samples
        max_lag (int): maximum lag

    Returns:
        float: negative of the estimate of autocorrelation time
    """
    n = len(xs)
    if max_lag is None:
        max_lag = n - 1
    if max_lag < 1:
        raise ValueError("max_lag needs to be greater than 0.")

    ac = 0.0
    seq = xs - np.mean(xs)
    var = np.var(seq)

    if np.isclose(var, 0.0):
        return -float(max_lag)

    for lag in range(1, max_lag+1):
        ac += np.abs(np.sum(seq[0:(n-lag)] * seq[lag:n]) / (n*var))
    return -ac


def efficiency_naive(target_distribution, x0, xs, accepteds, max_lag=None):
    return -1.0 / auto_correlation_naive(target_distribution, xs, accepteds, max_lag)


def auto_correlation_geyer(target_distribution, x0, xs, accepteds, max_lag=None):
    """
    Calculates an estimate of the autocorrelation time.
    Uses the initial monotone sequence estimator proposed in Geyer (1992).

    Parameters:
        xs (np.ndarray): samples
        max_lag (int): maximum lag

    Returns:
        float: negative of the estimate of autocorrelation time
    """
    n = len(xs)
    if max_lag is None:
        max_lag = n - 1
    if max_lag < 1:
        raise ValueError("max_lag needs to be greater than 0.")

    acorr = np.zeros(max_lag+1)
    pair_count = int(np.floor((max_lag + 1) / 2.0))
    pair_sums = np.zeros(pair_count)
    seq = xs - np.mean(xs)
    var = np.var(seq)

    if np.isclose(var, 0.0):
        return -2*n + 1.0

    for lag in range(0, max_lag+1):
        acorr[lag] = np.sum(seq[0:(n-lag)] * seq[lag:n]) / (n*var)
        if lag % 2 == 1:
            pair_sum = acorr[lag] + acorr[lag-1]
            # pair sums are always positive
            # if the sum of autocorrelations for adjacent lags is negative, we can stop
            if pair_sum < 0.0:
                break

            pair_ix = int(np.floor(lag / 2.0))
            # pair sums are decreasing
            # if the current pair sum is greater than the previous, set it to the previous.
            if pair_ix > 0 and pair_sum > pair_sums[pair_ix - 1]:
                pair_sum = pair_sums[pair_ix - 1]
            pair_sums[int(np.floor(lag / 2.0))] = pair_sum
    acorr_time = np.sum(2*pair_sums) - acorr[0]
    return -acorr_time


def efficiency_geyer(target_distribution, x0, xs, accepteds, max_lag=None):
    return -1.0 / auto_correlation_geyer(target_distribution, xs, accepteds, max_lag)


def auto_correlation_batch_means(target_distribution, x0, xs, accepteds, batch_count=4):
    """
    Calculates an estimate of the autocorrelation time using the batch_means method.
    See Geyer (1992) or Thompson (2010).

    Parameters:
        xs (np.ndarray): samples

    Returns:
        float: negative of the estimate of autocorrelation time
    """
    n = len(xs)
    batch_size = int(np.round(n / batch_count))
    samples = xs[0:(batch_count * batch_size)]
    batch_samples = np.reshape(samples, (batch_count, -1) + samples.shape[1:])
    var_batch_means = np.var(np.mean(batch_samples, axis=1), axis=0)
    var = np.var(samples, axis=0)
    if np.all(np.isclose(var, 0.0)):
        return -batch_size
    acorr_time = np.mean(batch_size * var_batch_means / var)
    return -acorr_time


def efficiency_batch_means(target_distribution, x0, xs, accepteds, batch_count=4):
    return -1.0 / auto_correlation_batch_means(target_distribution, xs, accepteds, batch_count)

