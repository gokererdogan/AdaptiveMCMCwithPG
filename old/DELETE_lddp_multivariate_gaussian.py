"""
Learning data-driven proposals through reinforcement learning

Learning to sample from a multivariate Gaussian

23 Nov. 2016
https://github.com/gokererdogan
"""
from gmllib.helpers import progress_bar

import warnings

import scipy.stats
import autograd.numpy as np
import autograd as ag


class ParallelMHChains(object):
    """
    ParallelMHChains class implements multiple Metropolis-Hastings chains running
    in parallel.
    """
    def __init__(self, target_distribution, policy, reward_function, chain_count=10, episode_length=40):
        self.target_distribution = target_distribution
        self.policy = policy
        self.reward_function = reward_function
        self.chain_count = chain_count
        self.episode_length = episode_length
        self.x0 = np.array([self.target_distribution.initial_x() for _ in range(self.chain_count)])

    def reset(self):
        self.x0 = np.array([self.target_distribution.initial_x() for _ in range(self.chain_count)])

    def log_acceptance_ratio(self, x, xp, params):
        """
        a(x -> xp)
        """
        logp_x = self.target_distribution.log_probability(x)
        logp_xp = self.target_distribution.log_probability(xp)
        log_q_xp_x = self.policy.log_propose_probability(x, xp, params)
        log_q_x_xp = self.policy.log_propose_probability(xp, x, params)
        log_a = (logp_xp + log_q_x_xp) - (logp_x + log_q_xp_x)
        return log_a

    def acceptance_ratio(self, x, xp):
        def _log_rr(pp):
            return np.log(1.0 - np.exp(self.log_acceptance_ratio(x, xp, pp)))

        g_logrr = ag.grad(_log_rr)
        ar = np.exp(self.log_acceptance_ratio(x, xp, self.policy.params))
        if np.isnan(ar):
            warnings.warn("Acceptance ratio is nan.")
        if ar < 1.0:
            drr = g_logrr(self.policy.params)
        else:
            drr = [np.zeros(p.shape) for p in self.policy.params]

        return ar, drr

    def run_episode(self):
        rewards = np.zeros(self.chain_count)
        xs = np.zeros((self.episode_length,) + self.x0[0].shape)
        accepteds = np.zeros(self.episode_length, dtype=np.bool)
        dlogps = [np.zeros((self.chain_count,) + p.shape) for p in self.policy.params]
        for c in range(self.chain_count):
            x = self.x0[c]
            for t in range(self.episode_length):
                xp = self.policy.propose(x)
                pp, dlog_p_xp_x, dlog_p_x_xp = self.policy.propose_probability(x, xp)
                a, dlog_r = self.acceptance_ratio(x, xp)
                if np.random.rand() < a:  # accept
                    accepteds[t] = True
                    x = xp

                    if a > 1.0:
                        gradient = dlog_p_xp_x
                    else:
                        gradient = dlog_p_x_xp
                else:
                    assert(not type(dlog_r) == float)
                    gradient = [t1+t2 for t1, t2 in zip(dlog_p_xp_x,  dlog_r)]

                # accumulate gradients
                for dl, gi in zip(dlogps, gradient):
                    dl[c] += gi

                # record current sample
                xs[t] = x

            # last sample for this episode will be the first sample in the next
            self.x0[c] = x
            rewards[c] = self.reward_function(self.target_distribution, xs, accepteds)
        return rewards, dlogps


class GradientEstimator(object):
    def __init__(self, clip=False):
        self.clip = clip
        pass

    def reset(self):
        pass

    def estimate_gradient(self, rewards, grad_logp):
        g = []
        for grad_logp_i in grad_logp:
            gi = np.mean((grad_logp_i.T * rewards).T, axis=0)
            if self.clip:
                if np.sum(np.square(gi)) > 1.0:
                    gi /= np.sum(np.square(gi))
            g.append(gi)
        return g


class BBVIEstimator(GradientEstimator):
    """
    Use score function as control variate
    Ranganath, R., Gerrish, S., & Blei, D. M. (2013). Black Box Variational Inference.
    """
    def __init__(self, clip=False):
        GradientEstimator.__init__(self, clip=clip)

    def estimate_gradient(self, rewards, grad_logp):
        cov = 0.0
        var = 0.0
        for grad_logp_i in grad_logp:
            dl = np.reshape(grad_logp_i, (grad_logp_i.shape[0], -1))
            dim = dl.shape[1]
            for i in range(dim):
                f = rewards * dl[:, i]
                g = dl[:, i]  # control variate
                cov_mat = np.cov(f, g)
                cov += cov_mat[0, 1]  # Cov(f, g)
                var += cov_mat[1, 1]  # Var(g)
        # calculate optimal scaling factor
        a_opt = cov / var

        rewards_baselined = rewards - a_opt
        g = GradientEstimator.estimate_gradient(self, rewards_baselined, grad_logp)
        return g


class VIMCOEstimator(GradientEstimator):
    def __init__(self, clip=False):
        GradientEstimator.__init__(self, clip=clip)

    def estimate_gradient(self, rewards, grad_logp):
        K = rewards.shape[0]
        tot_r = np.sum(rewards)
        rewards_baselined = (tot_r / K) - ((tot_r - rewards) / (K-1))
        g = GradientEstimator.estimate_gradient(self, rewards_baselined, grad_logp)
        return g


class TargetDistribution(object):
    def __init__(self):
        pass

    def initial_x(self):
        """
        Get initial state

        Returns:
            numpy.ndarray
        """
        raise NotImplementedError()

    def log_probability(self, x):
        raise NotImplementedError()

    def probability(self, x):
        """
        p(x)
        """
        raise NotImplementedError()


class MultivariateGaussian(TargetDistribution):
    def __init__(self, mean, cov):
        """
        Assumes mean and cov are numpy arrays.
        """
        TargetDistribution.__init__(self)
        self.mean = mean
        self.cov = cov
        self.D = self.mean.size
        
        self.dist = scipy.stats.multivariate_normal(mean=self.mean, cov=self.cov)

    def initial_x(self):
        """
        Get initial state

        Returns:
            numpy.ndarray
        """
        six_sigma = 6 * np.sqrt(np.diag(self.cov))
        return ((np.random.rand(self.D) - 0.5) * 2.0 * six_sigma) + self.mean

    def log_probability(self, x):
        return self.dist.logpdf(x)

    def probability(self, x):
        """
        p(x)
        """
        return self.dist.pdf(x)


class Policy(object):
    def __init__(self):
        pass

    def get_proposal_distribution(self, x, params):
        raise NotImplementedError()

    def propose(self, x):
        """
        xp ~ q(xp|x)
        """
        raise NotImplementedError()

    def log_propose_probability(self, x, xp, params):
        raise NotImplementedError()

    def propose_probability(self, x, xp):
        raise NotImplementedError()


class LinearGaussianPolicy(Policy):
    def __init__(self, D, mean=None, sd=None):
        Policy.__init__(self)
        self.D = D
        self.params = []
        if mean is None:
            self.mean_fixed = False
            # w
            self.params.append(np.random.randn(self.D, self.D) * 0.01)
            # b
            self.params.append(np.random.randn(self.D) * 0.01)
        else:
            self.mean_fixed = True
            self.mean = mean

        if sd is None:
            self.sd_fixed = False
            # w
            self.params.append(np.random.randn(self.D, self.D) * 0.01)
            # b
            self.params.append(np.random.randn(self.D) * 0.01)
        else:
            self.sd_fixed = True
            self.sd = sd

    def get_proposal_distribution(self, x, params):
        if self.mean_fixed:
            mean = self.mean
        else:
            w_m = params[0]
            b_m = params[1]
            mean = np.dot(x, w_m) + b_m

        if self.sd_fixed:
            sd = self.sd
        else:
            w_sd = params[-2]
            b_sd = params[-1]
            sd = np.exp(np.dot(x, w_sd) + b_sd)

        return mean, sd

    def propose(self, x):
        """
        xp ~ q(xp|x)
        """
        m, sd = self.get_proposal_distribution(x, self.params)
        a = m + sd*np.random.randn(m.size)
        xp = x + a
        return xp

    def log_propose_probability(self, x, xp, params):
        m_x, sd_x = self.get_proposal_distribution(x, params)
        q_xp_x = -0.5*(np.sum((xp - x - m_x)**2 / (sd_x**2))) - 0.5*self.D*np.log(2*np.pi) - np.sum(np.log(sd_x))
        return q_xp_x

    def propose_probability(self, x, xp):
        def _log_ppf(pp):
            return self.log_propose_probability(x, xp, pp)

        def _log_ppb(pp):
            return self.log_propose_probability(xp, x, pp)

        g_logppf = ag.grad(_log_ppf)
        g_logppb = ag.grad(_log_ppb)
        return np.exp(self.log_propose_probability(x, xp, self.params)), g_logppf(self.params), g_logppb(self.params)


class NonlinearGaussianPolicy(Policy):
    def __init__(self, D, n_hidden, mean=None, sd=None):
        Policy.__init__(self)
        self.D = D
        self.n_hidden = n_hidden
        self.params = []
        # initialize input-hidden weight matrix
        self.w_input_hidden = np.random.randn(self.D, self.n_hidden) * 0.01
        self.b_hidden = np.random.randn(self.n_hidden) * 0.01
        self.params.append(self.w_input_hidden)
        self.params.append(self.b_hidden)
        if mean is None:
            self.mean_fixed = False
            # hidden to output w
            self.w_mean = np.random.randn(self.n_hidden, self.D) * 0.01
            self.params.append(self.w_mean)
            # hidden to output b
            self.b_mean = np.random.randn(self.D) * 0.01
            self.params.append(self.b_mean)
        else:
            self.mean_fixed = True
            self.mean = mean

        if sd is None:
            self.sd_fixed = False
            # hidden to output w
            self.w_sd = np.random.randn(self.n_hidden, self.D) * 0.01
            self.params.append(self.w_sd)
            # hidden to output b
            self.b_sd = np.random.randn(self.D) * 0.01
            self.params.append(self.b_sd)
        else:
            self.sd_fixed = True
            self.sd = sd

    def get_proposal_distribution(self, x, params):
        w_input_hidden = params[0]
        b_hidden = params[1]
        hidden_activations = np.tanh(np.dot(x, w_input_hidden) + b_hidden)
        if self.mean_fixed:
            mean = self.mean
        else:
            w_mean = params[2]
            b_mean = params[3]
            mean = np.dot(hidden_activations, w_mean) + b_mean

        if self.sd_fixed:
            sd = self.sd
        else:
            w_sd = params[-2]
            b_sd = params[-1]
            sd = np.exp(np.dot(hidden_activations, w_sd)) + b_sd

        return mean, sd

    def propose(self, x):
        """
        xp ~ q(xp|x)
        """
        m, sd = self.get_proposal_distribution(x, self.params)
        a = m + sd*np.random.randn(m.size)
        xp = x + a
        return xp

    def log_propose_probability(self, x, xp, params):
        m_x, sd_x = self.get_proposal_distribution(x, params)
        q_xp_x = -0.5*(np.sum((xp - x - m_x)**2 / (sd_x**2))) - 0.5*self.D*np.log(2*np.pi) - np.sum(np.log(sd_x))
        return q_xp_x

    def propose_probability(self, x, xp):
        def _log_ppf(pp):
            return self.log_propose_probability(x, xp, pp)

        def _log_ppb(pp):
            return self.log_propose_probability(xp, x, pp)

        g_logppf = ag.grad(_log_ppf)
        g_logppb = ag.grad(_log_ppb)
        return np.exp(self.log_propose_probability(x, xp, self.params)), g_logppf(self.params), g_logppb(self.params)


"""
PARAMETER SCHEDULE
"""


class ParameterSchedule(object):
    def __init__(self):
        pass

    def get_value(self, iteration_no):
        raise NotImplementedError()


class ConstantSchedule(ParameterSchedule):
    def __init__(self, value):
        self.value = value

    def get_value(self, iteration_no):
        return self.value

    def __str__(self):
        return "Constant schedule, value: {0:f}".format(self.value)

    def __repr__(self):
        return self.__str__()


class LinearSchedule(ParameterSchedule):
    def __init__(self, start_value, end_value, decrease_for):
        ParameterSchedule.__init__(self)
        self.start_value = start_value
        self.end_value = end_value
        self.decrease_for = decrease_for
        self.decreasing = self.start_value > self.end_value

    def get_value(self, iteration_no):
        assert(iteration_no >= 0)
        value = self.start_value + (iteration_no * (self.end_value - self.start_value) / self.decrease_for)
        if (self.decreasing and value < self.end_value) or (not self.decreasing and value > self.end_value):
            value = self.end_value
        return value

    def __str__(self):
        return "Linear schedule, start: {0:f}, end: {1:f}, decrease for {2:d}".format(self.start_value,
                                                                                      self.end_value,
                                                                                      self.decrease_for)

    def __repr__(self):
        return self.__str__()

"""
REWARD FUNCTIONS
"""


def reward_log_prob(target_distribution, xs, accepteds):
    """
    Returns average log probability of samples xs.
    """
    return np.mean([target_distribution.log_probability(x) for x in xs])


def reward_acceptance_rate(target_distribution, xs, accepteds):
    """
    Returns acceptance rate for samples xs.
    """
    return np.mean(accepteds)


def reward_auto_correlation_naive(target_distribution, xs, accepteds, max_lag=None):
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


def reward_efficiency_naive(target_distribution, xs, accepteds, max_lag=None):
    return -1.0 / reward_auto_correlation_naive(target_distribution, xs, accepteds, max_lag)


def reward_auto_correlation_geyer(target_distribution, xs, accepteds, max_lag=None):
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


def reward_efficiency_geyer(target_distribution, xs, accepteds, max_lag=None):
    return -1.0 / reward_auto_correlation_geyer(target_distribution, xs, accepteds, max_lag)


def reward_auto_correlation_batch_means(target_distribution, xs, accepteds, batch_count=4):
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


def reward_efficiency_batch_means(target_distribution, xs, accepteds, batch_count=4):
    return -1.0 / reward_auto_correlation_batch_means(target_distribution, xs, accepteds, batch_count)


"""OPTIMIZATION FUNCTIONS"""


def adam(chains, gradient_estimator, learning_rate_schedule, epoch_count=100,
         episodes_per_epoch=10, report_period=10, b1=0.9, b2=0.999, eps=10**-8):
    """
    Gradient ascent with adam.
    Taken from autograd's adam implementation
    """
    """
    # adam parameters
    b1 = 0.9
    b2 = 0.999
    eps = 10**-8
    """

    param_count = len(chains.policy.params)
    max_iteration = epoch_count * episodes_per_epoch
    rewards = np.zeros(max_iteration)
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
            progress_bar(iteration+1, max_iteration, update_freq=max_iteration/100 or 1)
            rs, dps = chains.run_episode()
            rewards[iteration] = np.mean(rs)

            g = gradient_estimator.estimate_gradient(rs, dps)
            for i, gi in enumerate(g):
                m[i] = (1 - b1) * gi + b1 * m[i]  # First  moment estimate.
                v[i] = (1 - b2) * (gi**2) + b2 * v[i]  # Second moment estimate.
                mhat = m[i] / (1 - b1**(iteration + 1))    # Bias correction.
                vhat = v[i] / (1 - b2**(iteration + 1))
                grad_magnitudes[iteration, i] = np.sum(np.square(gi))
                learning_rate = learning_rate_schedule.get_value(iteration)
                chains.policy.params[i] += learning_rate * (mhat / (np.sqrt(vhat) + eps))

        if (epoch+1) % report_period == 0:
            print
            print "Epoch {0:d}\tAvg. Reward: {1:.3f}".format(epoch+1, np.mean(rs))

        # store params
        ps = []
        for p in chains.policy.params:
            ps.append(p.copy())
        params.append(ps)

        # reset the chains (i.e., start from random states)
        chains.reset()

    return params, rewards, grad_magnitudes


"""MISC. FUNCTIONS"""


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
    # np.seterr(all='raise')
    # np.random.seed(1)

    """
    # vanilla_estimator = GradientEstimator()
    # bbvi_estimator = BBVIEstimator()
    vimco_estimator = VIMCOEstimator()

    my_policy = GaussianPolicy(D=2, mean=np.zeros(2))
    my_target = MultivariateGaussian(mean=np.zeros(2), cov=np.eye(2))
    my_chains = ParallelMHChains(target_distribution=my_target, policy=my_policy,
                                 reward_function=reward_auto_correlation, chain_count=10,
                                 episode_length=40)

    adam(my_chains, vimco_estimator, learning_rate=0.01, epoch_count=10, episodes_per_epoch=25, report_period=1)
    """
    my_target = MultivariateGaussian(mean=0.0, cov=1.0)
    my_policy = LinearGaussianPolicy(D=1, mean=np.zeros(1))
    my_chains = ParallelMHChains(target_distribution=my_target, policy=my_policy,
                                 reward_function=reward_auto_correlation_batch_means, chain_count=20,
                                 episode_length=48)

    my_rewards = estimate_reward_surface_b(my_chains, seed=1)
