"""
Learning data-driven proposals through reinforcement learning

This file contains the classes for representing policies, i.e., proposal distributions.

13 Dec. 2016
https://github.com/gokererdogan
"""
import autograd.numpy as np
import autograd as ag


class Policy(object):
    def __init__(self, seed=None):
        if seed is None:
            np.random.seed()
        else:
            np.random.seed(seed)
        # function for calculating the gradient of log propose probability
        self.grad_log_propose_probability = ag.grad(self.log_propose_probability, 3)

    def __str__(self):
        return "Abstract Policy class"

    def __repr__(self):
        return self.__str__()

    def get_proposal_distribution(self, x, data, params):
        # this method is not used by sampler (and thus optional).
        # it is just nice to split the proposal into two separate steps:
        #  1) getting the distribution and 2) sampling from it.
        pass

    def propose(self, x, data):
        """
        xp ~ q(xp|x)
        """
        raise NotImplementedError()

    def log_propose_probability(self, x, data, xp, params):
        raise NotImplementedError()

    def __getstate__(self):
        # functions cannot be pickled, so get rid of it.
        # Why need pickle?
        #   Experiment instances are pickled which in turn pickle experiment parameters, which may be Policy instances.
        d = self.__dict__.copy()
        del d['grad_log_propose_probability']
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        # recreate the gradient function
        self.grad_log_propose_probability = ag.grad(self.log_propose_probability, 3)


class LinearGaussianPolicy(Policy):
    def __init__(self, D, mean=None, sd=None, seed=None):
        Policy.__init__(self, seed=seed)
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

    def __str__(self):
        mean = self.mean if self.mean_fixed else None
        sd = self.sd if self.sd_fixed else None
        return "LinearGaussianPolicy with D={0:d}, mean={1:s} and sd={2:s}".format(self.D, mean, sd)

    def __repr__(self):
        return self.__str__()

    def get_proposal_distribution(self, x, data, params):
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

    def propose(self, x, data):
        """
        xp ~ q(xp|x)
        """
        m, sd = self.get_proposal_distribution(x, data, self.params)
        a = m + sd*np.random.randn(m.size)
        xp = x + a
        return xp

    def log_propose_probability(self, x, data, xp, params):
        m_x, sd_x = self.get_proposal_distribution(x, data, params)
        q_xp_x = -0.5*(np.sum((xp - x - m_x)**2 / (sd_x**2))) - 0.5*self.D*np.log(2*np.pi) - np.sum(np.log(sd_x))
        return q_xp_x


class NonlinearGaussianPolicy(Policy):
    def __init__(self, D, n_hidden, mean=None, sd=None, seed=None):
        Policy.__init__(self, seed=seed)
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

    def __str__(self):
        mean = self.mean if self.mean_fixed else None
        sd = self.sd if self.sd_fixed else None
        return "NonlinearGaussianPolicy with D={0:d}, n_hidden={1:d}, " \
               "mean={2:s} and sd={3:s}".format(self.D, self.n_hidden, mean, sd)

    def __repr__(self):
        return self.__str__()

    def get_proposal_distribution(self, x, data, params):
        w_input_hidden = params[0]
        b_hidden = params[1]
        hidden_activations = np.maximum(np.dot(x, w_input_hidden) + b_hidden, 0.0)
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

    def propose(self, x, data):
        """
        xp ~ q(xp|x)
        """
        m, sd = self.get_proposal_distribution(x, data, self.params)
        a = m + sd*np.random.randn(m.size)
        xp = x + a
        return xp

    def log_propose_probability(self, x, data, xp, params):
        m_x, sd_x = self.get_proposal_distribution(x, data, params)
        q_xp_x = -0.5*(np.sum((xp - x - m_x)**2 / (sd_x**2))) - 0.5*self.D*np.log(2*np.pi) - np.sum(np.log(sd_x))
        return q_xp_x

