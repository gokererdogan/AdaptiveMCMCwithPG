"""
Learning data-driven proposals through reinforcement learning

This file contains the distribution and policy classes for 2D find the dot problem.

16 Dec. 2016
https://github.com/gokererdogan
"""
import autograd.numpy as np
import autograd as ag

from target_distribution import TargetDistribution
from policy import Policy

IMG_SIZE = 9
X_BOUND = 4.0
XS = np.linspace(-X_BOUND, X_BOUND, IMG_SIZE+1) + (X_BOUND / IMG_SIZE)
XS = XS[0:IMG_SIZE]
OUT_OF_BOUNDS_PENALTY = 20


class FindTheDotTargetDistribution(TargetDistribution):
    def __init__(self, ll_variance, data=None):
        TargetDistribution.__init__(self)
        self.ll_variance = ll_variance
        self.data = data
        if self.data is None:
            self.data = FindTheDotTargetDistribution.render(self.initial_x())

    def __str__(self):
        return "FindTheDotTargetDistribution with IMG_SIZE={0}".format(IMG_SIZE)

    def __repr__(self):
        return self.__str__()

    def initial_x(self):
        return np.random.choice(XS)

    def reset(self):
        self.data = FindTheDotTargetDistribution.render(self.initial_x())

    @staticmethod
    def render(x):
        img = np.zeros(IMG_SIZE)
        if np.abs(x) < X_BOUND:
            pos = np.argmin(np.abs(XS - x))
            img[pos] = 1.0
        return img

    def log_probability(self, x):
        out_of_bounds = np.abs(x) > X_BOUND
        render = FindTheDotTargetDistribution.render(x)
        diff = np.sum(np.square((render - self.data))) / self.ll_variance
        if out_of_bounds:
            diff += OUT_OF_BOUNDS_PENALTY
        return -0.5*diff

    def probability(self, x):
        return np.exp(self.log_probability(x))


class FindTheDotPolicy(Policy):
    def __init__(self, type='linear', n_hidden=10):
        Policy.__init__(self)
        self.type = type
        self.input_size = IMG_SIZE
        self.params = []
        if self.type == 'linear':
            self.params.append(np.random.randn(self.input_size, 1) * 0.01)
        elif self.type == 'nonlinear':
            self.n_hidden = n_hidden
            # w_h
            self.params.append(np.random.randn(self.input_size, self.n_hidden) * 0.01)
            # b_h
            self.params.append(np.random.randn(self.n_hidden) * 0.01)
            # w_m
            self.params.append(np.random.randn(self.n_hidden, 1) * 0.01)
            # w_sd
            self.params.append(np.random.randn(self.n_hidden, 1) * 0.01)
        else:
            raise ValueError("Wrong policy type.")

    def __str__(self):
        s = "FindTheDotPolicy with type={0}".format(self.type)
        if self.type == 'nonlinear':
            s += " n_hidden={0}".format(self.n_hidden)
        return s

    def __repr__(self):
        return self.__str__()

    def get_proposal_distribution(self, x, data, params):
        img = FindTheDotTargetDistribution.render(x)
        nn_input = np.ravel(img - data)
        if self.type == 'linear':
            w = params[0]
            mean = np.dot(nn_input, w)
            sd = 1.0
        elif self.type == 'nonlinear':
            w_h = params[0]
            b_h = params[1]
            hidden_activations = np.tanh(np.dot(nn_input, w_h) + b_h)

            w_m = params[2]
            mean = np.dot(hidden_activations, w_m)

            w_sd = params[3]
            sd = np.exp(np.dot(hidden_activations, w_sd))

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
        q_xp_x = -0.5*(np.sum((xp - x - m_x)**2 / (sd_x**2))) - 0.5*1*np.log(2*np.pi) - np.sum(np.log(sd_x))
        return q_xp_x

    def propose_probability(self, x, data, xp):
        def _log_ppf(pp):
            return self.log_propose_probability(x, data, xp, pp)

        def _log_ppb(pp):
            return self.log_propose_probability(xp, data, x, pp)

        g_logppf = ag.grad(_log_ppf)
        g_logppb = ag.grad(_log_ppb)
        return np.exp(self.log_propose_probability(x, data, xp, self.params)), g_logppf(self.params), g_logppb(self.params)

