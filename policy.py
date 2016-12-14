"""
Learning data-driven proposals through reinforcement learning

This file contains the classes for representing policies, i.e., proposal distributions.

13 Dec. 2016
https://github.com/gokererdogan
"""
import autograd.numpy as np
import autograd as ag


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

