"""
Learning data-driven proposals through reinforcement learning

This file contains the distribution and policy classes for 2D find the square problem.

15 Dec. 2016
https://github.com/gokererdogan
"""
import autograd.numpy as np
import autograd as ag

from target_distribution import TargetDistribution
from policy import Policy

SQUARE_SIZE = (7, 7)
CANVAS_SIZE = (19, 19)
# Image is larger than canvas (drawing area) to ensure that the square always stays fully within
# the image.
IMG_SIZE = (CANVAS_SIZE[0] + 2*(SQUARE_SIZE[0]/2), CANVAS_SIZE[1] + 2*(SQUARE_SIZE[1]/2))
X_BOUND = 6.0
Y_BOUND = 6.0
XS = np.linspace(-X_BOUND, X_BOUND, CANVAS_SIZE[0]+1) + (X_BOUND / CANVAS_SIZE[0])
XS = XS[0:CANVAS_SIZE[0]]
YS = np.linspace(-Y_BOUND, Y_BOUND, CANVAS_SIZE[1]+1) + (Y_BOUND / CANVAS_SIZE[1])
YS = YS[0:CANVAS_SIZE[1]]


class FindTheSquareTargetDistribution(TargetDistribution):
    def __init__(self, ll_variance, data=None):
        TargetDistribution.__init__(self)
        self.ll_variance = ll_variance
        self.data = data
        if self.data is None:
            self.data = FindTheSquareTargetDistribution.render(self.initial_x())

    def __str__(self):
        return "FindTheSquareTargetDistribution with CANVAS_SIZE={0}, " \
               "SQUARE_SIZE={1}, ll variance={2}".format(CANVAS_SIZE, SQUARE_SIZE, self.ll_variance)

    def __repr__(self):
        return self.__str__()

    def initial_x(self):
        return np.array([np.random.choice(XS), np.random.choice(YS)])

    def reset(self):
        self.data = FindTheSquareTargetDistribution.render(self.initial_x())

    @staticmethod
    def render(x):
        img = np.zeros(IMG_SIZE)
        if np.abs(x[0]) < X_BOUND and np.abs(x[1]) < Y_BOUND:
            w, h = SQUARE_SIZE
            posx = np.argmin(np.abs(XS - x[0])) + w/2
            posy = np.argmin(np.abs(YS - x[1])) + h/2
            xs = posx - w/2
            assert(xs >= 0)
            xe = xs + w
            assert(xe <= IMG_SIZE[0])
            ys = posy - h/2
            assert(ys >= 0)
            ye = ys + h
            assert(ye <= IMG_SIZE[1])
            img[xs:xe, ys:ye] = 1.0
        return img

    def log_probability(self, x):
        # count the number of squares that are out of bounds
        out_of_bounds = np.abs(x[0]) > X_BOUND or np.abs(x[1]) > Y_BOUND
        render = FindTheSquareTargetDistribution.render(x)
        diff = np.sum(np.square((render - self.data))) / self.ll_variance
        out_of_bounds_penalty = 2 * SQUARE_SIZE[0] * SQUARE_SIZE[1] / self.ll_variance
        if out_of_bounds:
            diff += out_of_bounds_penalty
        return -0.5*diff

    def probability(self, x):
        return np.exp(self.log_probability(x))


class FindTheSquarePolicy(Policy):
    def __init__(self, n_hidden=None):
        Policy.__init__(self)
        self.input_size = IMG_SIZE[0] * IMG_SIZE[1]
        self.params = []
        self.n_hidden = n_hidden
        # w_h
        self.params.append(np.random.randn(self.input_size, self.n_hidden) * 0.01)
        # b_h
        self.params.append(np.random.randn(self.n_hidden) * 0.01)
        # w_m
        self.params.append(np.random.randn(self.n_hidden, 2) * 0.01)
        # w_sd
        self.params.append(np.random.randn(self.n_hidden, 2) * 0.01)

    def __str__(self):
        return "FindTheSquarePolicy with n_hidden={0}".format(self.n_hidden)

    def __repr__(self):
        return self.__str__()

    def get_proposal_distribution(self, x, data, params):
        img = FindTheSquareTargetDistribution.render(x)
        nn_input = np.ravel(img - data)

        w_h = params[0]
        b_h = params[1]
        hidden_activations = np.tanh(np.dot(nn_input, w_h) + b_h)

        w_m = params[2]
        m = np.dot(hidden_activations, w_m)

        w_sd = params[3]
        sd = np.exp(np.dot(hidden_activations, w_sd))

        return m, sd

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
        q_xp_x = -0.5*(np.sum((xp - x - m_x)**2 / (sd_x**2))) - 0.5*2*np.log(2*np.pi) - np.sum(np.log(sd_x))
        return q_xp_x

    def propose_probability(self, x, data, xp):
        def _log_ppf(pp):
            return self.log_propose_probability(x, data, xp, pp)

        def _log_ppb(pp):
            return self.log_propose_probability(xp, data, x, pp)

        g_logppf = ag.grad(_log_ppf)
        g_logppb = ag.grad(_log_ppb)
        return np.exp(self.log_propose_probability(x, data, xp, self.params)), g_logppf(self.params), g_logppb(self.params)

