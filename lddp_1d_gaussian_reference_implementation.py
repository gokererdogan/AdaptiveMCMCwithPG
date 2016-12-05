from gmllib.helpers import progress_bar

import autograd.numpy as np
from autograd import grad


def initial_x():
    return (np.random.rand()*12.0) - 6.0


def reward(x):
    """
    R(x)
    """
    return log_probability(x)


def auto_correlation(xs, max_lag=None):
    n = len(xs)
    if max_lag is None:
        max_lag = n-1
    if max_lag < 1:
        raise ValueError("max_lag needs to be greater than 0.")

    ac = np.zeros(max_lag)
    seq = xs - np.mean(xs)
    var = np.var(seq)

    if np.isclose(var, 0.0):
        return np.ones(max_lag)

    for lag in range(1, max_lag+1):
        ac[lag-1] = np.abs(np.sum(seq[0:(n-lag)] * seq[lag:n]) / (n*var))
    return ac


def log_probability(x):
    return -0.5*x*x - np.log(np.sqrt(2*np.pi))


def probability(x):
    """
    p(x)
    """
    return np.exp(log_probability(x))


def propose(x, w, b):
    """
    xp ~ q(xp|x)
    """
    sd = np.exp(w*x + b)
    a = 0.0 + sd*np.random.randn()
    xp = x + a
    return xp


def log_propose_probability(x, xp, w, b):
    sd_x = np.exp(w*x + b)
    q_xp_x = -0.5*((xp-x)**2 / (sd_x**2)) - np.log((np.sqrt(2*np.pi) * sd_x))
    return q_xp_x


def propose_probability(x, xp, w, b):
    def _log_ppf(params):
        return log_propose_probability(x, xp, params[0], params[1])

    def _log_ppb(params):
        return log_propose_probability(xp, x, params[0], params[1])

    g_logppf = grad(_log_ppf)
    g_logppb = grad(_log_ppb)
    return np.exp(log_propose_probability(x, xp, w, b)), g_logppf(np.array([w, b])), g_logppb(np.array([w, b]))


def log_acceptance_ratio(x, xp, w, b):
    """
    a(x -> xp)
    """
    logp_x = log_probability(x)
    logp_xp = log_probability(xp)
    log_q_xp_x = log_propose_probability(x, xp, w, b)
    log_q_x_xp = log_propose_probability(xp, x, w, b)
    log_a = (logp_xp + log_q_x_xp) - (logp_x + log_q_xp_x)
    return log_a


def acceptance_ratio(x, xp, w, b):
    def _log_rr(params):
        return np.log(1.0 - np.exp(log_acceptance_ratio(x, xp, params[0], params[1])))

    g_logrr = grad(_log_rr)
    ar = np.exp(log_acceptance_ratio(x, xp, w, b))
    params = np.array([w, b])
    drr = 0.0
    if ar < 1.0:
        drr = g_logrr(params)

    return ar, drr


def run_episode(x0, w, b, runs, episode_length):
    rewards = np.zeros(runs)
    acceptance_rates = np.zeros(runs)
    autocorrelations = np.zeros(runs)
    dlogps = np.zeros((runs, 2))
    xs = np.zeros(episode_length)
    for r in range(runs):
        x = x0[r]
        rewards[r] += reward(x)
        for t in range(episode_length):
            xp = propose(x, w, b)
            pp, dlog_pxpx, dlog_pxxp = propose_probability(x, xp, w, b)
            a, dlogr = acceptance_ratio(x, xp, w, b)
            if np.random.rand() < a:  # accept
                acceptance_rates[r] += 1
                x = xp

                if a > 1.0:
                    dlogps[r] += dlog_pxpx
                else:
                    dlogps[r] += dlog_pxxp
            else:
                dlogps[r] += (dlog_pxpx + dlogr)

            xs[t] = x
            rewards[r] += reward(x)
        x0[r] = x
        rewards[r] /= episode_length
        acceptance_rates[r] /= episode_length
        autocorrelations[r] += (np.sum(np.abs(auto_correlation(xs, lag_step=1))) / episode_length)
    return x0, rewards, acceptance_rates, autocorrelations, dlogps


def policy_gradient(rewards, dlogps):
    g = np.dot(rewards, dlogps) / rewards.shape[0]
    return g


def policy_gradient_bbvi(rewards, dlogps):
    """
    Use score function as control variate
    Ranganath, R., Gerrish, S., & Blei, D. M. (2013). Black Box Variational Inference.
    """
    dim = dlogps.shape[1]
    covs = np.zeros(dim)
    vars = np.zeros(dim)
    for i in range(dim):
        f = rewards * dlogps[:, i]
        g = dlogps[:, i]  # control variate
        cov = np.cov(f, g)
        covs[i] = cov[0, 1]  # Cov(f, g)
        vars[i] = cov[1, 1]  # Var(g)
    a_opt = np.sum(covs) / np.sum(vars)
    g = np.dot((rewards - a_opt), dlogps) / rewards.shape[0]
    return g


def policy_gradient_vimco_mean(rewards, dlogps):
    K = rewards.shape[0]
    tot_r = np.sum(rewards)
    rewards_baselined = (tot_r / K) - ((tot_r - rewards) / (K-1))
    g = np.dot(rewards_baselined, dlogps)
    return g


def adam(gradient_estimator, w, b, runs=100, episode_length=10, learning_rate=0.001, epoch_count=100,
         episodes_per_epoch=10, report_period=10):
    # adam parameters
    b1 = 0.9
    b2 = 0.999
    eps = 10**-8

    max_iteration = epoch_count * episodes_per_epoch
    m = np.zeros(2)
    v = np.zeros(2)
    for epoch in range(epoch_count):
        x0 = np.array([initial_x() for _ in range(runs)])
        for episode in range(episodes_per_epoch):
            iteration = (epoch * episodes_per_epoch) + episode
            progress_bar(iteration+1, max_iteration, update_freq=max_iteration/100 or 1)

            x0, rs, ars, acs, dps = run_episode(x0, w, b, runs=runs, episode_length=episode_length)

            g = gradient_estimator(acs, dps)
            # taken from autograd's adam implementation
            m = (1 - b1) * g      + b1 * m  # First  moment estimate.
            v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
            mhat = m / (1 - b1**(epoch + 1))    # Bias correction.
            vhat = v / (1 - b2**(epoch + 1))
            w -= learning_rate*mhat[0]/(np.sqrt(vhat[0] + eps))
            b -= learning_rate*mhat[1]/(np.sqrt(vhat[1] + eps))

        if (epoch+1) % report_period == 0:
            print
            print "Epoch", epoch+1, w, b
            print np.mean(rs), np.mean(ars), np.mean(acs)

    return w, b

if __name__ == "__main__":
    np.random.seed(1)
    # np.seterr(all='raise')
    w0 = np.random.randn() * 0.01
    b0 = np.random.randn() * 0.01
    print w0, b0
    w0, b0 = adam(policy_gradient_vimco_mean, w0, b0, learning_rate=0.01, runs=10, episode_length=40,
                  episodes_per_epoch=10, epoch_count=10, report_period=1)
