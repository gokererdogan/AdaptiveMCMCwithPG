"""
Learning data-driven proposals through reinforcement learning

Occluding tiles problem. Taken from
    Jampani, V., Nowozin, S., Loper, M., & Gehler, P. V. (2014). The Informed Sampler: A Discriminative Approach
        to Bayesian Inference in Generative Computer Vision Models. arXiv 1402.0859.

9 Jan. 2016
https://github.com/gokererdogan
"""
import skimage.draw as draw

import autograd.numpy as np
import autograd as ag

from target_distribution import TargetDistribution
from policy import Policy

MIN_TILE_SIZE = (10, 10)
MAX_TILE_SIZE = (30, 30)
IMG_SIZE = (50, 50)
X_BOUND = 6.0
COLORS = [(0.969, 0.076, 0.076), (0.995, 0.995, 0.161), (0.141, 0.978, 0.141),
          (0.165, 0.960, 0.960), (0.076, 0.076, 0.988), (0.981, 0.162, 0.981)]
OUT_OF_BOUNDS_PENALTY = 10e6

from collections import OrderedDict
class RenderCache(object):
    def __init__(self):
        self.cache_size = 100
        self.cache = OrderedDict()

    def in_cache(self, x):
        return tuple(x) in self.cache

    def get(self, x):
        return self.cache[tuple(x)]

    def add(self, x, img):
        if len(self.cache) >= self.cache_size:
            self.cache.popitem(last=False)
        self.cache[tuple(x)] = img

render_cache = RenderCache()


class OccludingTilesDistribution(TargetDistribution):
    def __init__(self, ll_sd, data=None):
        TargetDistribution.__init__(self)
        self.ll_sd = ll_sd
        self.data = data
        if self.data is None:
            self.true_x = self.initial_x()
            self.data = OccludingTilesDistribution.render(self.true_x)

    def __str__(self):
        return "OccludingTilesDistribution with IMG_SIZE={0}, " \
               "ll sd={1}".format(IMG_SIZE, self.ll_sd)

    def __repr__(self):
        return self.__str__()

    def initial_x(self):
        return (np.random.rand(24) - 0.5) * 2.0 * X_BOUND

    def reset(self):
        self.true_x = self.initial_x()
        self.data = OccludingTilesDistribution.render(self.true_x)

    @staticmethod
    def draw_tile(tile, img, color):
        cx, cy, z, theta = tile
        cx = ((cx + X_BOUND) / (2.0 * X_BOUND)) * IMG_SIZE[0]
        cy = ((cy + X_BOUND) / (2.0 * X_BOUND)) * IMG_SIZE[1]
        w = ((MAX_TILE_SIZE[0] - MIN_TILE_SIZE[0]) * (z + X_BOUND) / (2.0 * X_BOUND)) + MIN_TILE_SIZE[0]
        h = ((MAX_TILE_SIZE[1] - MIN_TILE_SIZE[1]) * (z + X_BOUND) / (2.0 * X_BOUND)) + MIN_TILE_SIZE[1]
        theta = -(np.pi / 4.0) + (np.pi * (theta + X_BOUND) / (4.0 * X_BOUND))

        # calculate position of corners by rotating the corners of an upright rectangle.
        x, y = -w/2, -h/2  # left top corner
        lt_x = cx + x * np.cos(theta) - y * np.sin(theta)
        lt_y = cy + x * np.sin(theta) + y * np.cos(theta)

        x, y = w/2, -h/2  # right top corner
        rt_x = cx + x * np.cos(theta) - y * np.sin(theta)
        rt_y = cy + x * np.sin(theta) + y * np.cos(theta)

        x, y = w/2, h/2  # right bottom corner
        rb_x = cx + x * np.cos(theta) - y * np.sin(theta)
        rb_y = cy + x * np.sin(theta) + y * np.cos(theta)

        x, y = -w/2, h/2  # left bottom corner
        lb_x = cx + x * np.cos(theta) - y * np.sin(theta)
        lb_y = cy + x * np.sin(theta) + y * np.cos(theta)

        poly_x = np.array([lt_x, rt_x, rb_x, lb_x, lt_x])
        poly_y = np.array([lt_y, rt_y, rb_y, lb_y, lt_y])
        filled_rows, filled_cols = draw.polygon(y=poly_y, x=poly_x, shape=IMG_SIZE)
        img[filled_rows, filled_cols] = color

    @staticmethod
    def render(x):
        if render_cache.in_cache(x):
            return render_cache.get(x)

        img = np.zeros(IMG_SIZE + (3,))
        # sort tiles according to depth. start from the tile that is farthest away.
        zs = x[2::4]  # get every 3rd element
        ix = np.argsort(zs)
        for i in ix:
            OccludingTilesDistribution.draw_tile(x[i*4:(i+1)*4], img, COLORS[i])

        render_cache.add(x, img)
        return img

    def log_probability(self, x):
        if np.any(np.abs(x) > X_BOUND):  # out of bounds
            return -OUT_OF_BOUNDS_PENALTY

        render = OccludingTilesDistribution.render(x)
        diff = np.mean(np.square((render - self.data) / self.ll_sd))
        return -0.5*diff

    def probability(self, x):
        return np.exp(self.log_probability(x))


class OccludingTilesPolicy(Policy):
    def __init__(self, sd_multiplier=1.0, n_hidden=100):
        Policy.__init__(self)
        self.pick_tile_network_input_size = IMG_SIZE[0] * IMG_SIZE[1] * 3
        self.move_tile_network_input_size = IMG_SIZE[0] * IMG_SIZE[1]
        self.sd_multiplier = sd_multiplier
        self.move_tile_network_n_hidden = n_hidden

        # pick tile network params
        self.params = []
        self.params.append(np.random.randn(self.pick_tile_network_input_size, 6) * 0.001)

        # move tile network params
        # w_h
        self.params.append(np.random.randn(self.move_tile_network_input_size, self.move_tile_network_n_hidden) * 0.01)
        # b_h
        self.params.append(np.random.randn(self.move_tile_network_n_hidden) * 0.01)
        # w_m
        self.params.append(np.random.randn(self.move_tile_network_n_hidden, 4) * 0.01)
        # w_sd
        self.params.append(np.random.randn(self.move_tile_network_n_hidden, 4) * 0.01)

    def __str__(self):
        return "OccludingTilesPolicy with sd_multiplier={0}, n_hidden={1}".format(self.sd_multiplier,
                                                                                  self.move_tile_network_n_hidden)

    def __repr__(self):
        return self.__str__()

    def get_proposal_distribution(self, x, data, params):
        # proposal consists of two stages and is implemented in the below two methods.
        pass

    def get_pick_tile_proposal(self, x, data, params):
        img = OccludingTilesDistribution.render(x)
        nn_input = np.ravel(img - data)

        # pick tile network
        w = params[0]
        pick_tile_outputs = np.dot(nn_input, w)
        pick_tile_probs = np.exp(pick_tile_outputs) / np.sum(np.exp(pick_tile_outputs))

        return pick_tile_probs

    def get_move_tile_proposal(self, x, data, tile, params):
        img = OccludingTilesDistribution.render(x)
        color = COLORS[tile]
        # pick the pixels with this color (mask all other tiles)
        mimg = np.zeros(IMG_SIZE)
        mimg[np.all(img == color, axis=2)] = 1.0
        mdata = np.zeros(IMG_SIZE)
        mdata[np.all(data == color, axis=2)] = 1.0
        nn_input = np.ravel(mimg - mdata)
        # move tile network
        # convert input to gray-scale
        w_h = params[1]
        b_h = params[2]
        hidden_activations = np.tanh(np.dot(nn_input, w_h) + b_h)

        w_m = params[3]
        move_mean = np.dot(hidden_activations, w_m)

        w_sd = params[4]
        move_sd = np.exp(np.dot(hidden_activations, w_sd))
        return move_mean, move_sd

    def propose(self, x, data):
        p_i = self.get_pick_tile_proposal(x, data, self.params)
        i = np.random.choice(6, p=p_i)
        mean, sd = self.get_move_tile_proposal(x, data, i, self.params)
        a = mean + sd*np.random.randn(mean.size)
        xp = x.copy()
        xp[i*4:(i+1)*4] += a
        # theta (orientation) is periodic. It wraps around -pi/4 when it exceeds pi/4 and vice versa
        theta_index = (i*4) + 3
        xp[theta_index] = ((xp[theta_index] + X_BOUND) % (2.0 * X_BOUND)) - X_BOUND
        return xp

    def log_propose_probability(self, x, data, xp, params):
        changed_elements = np.nonzero(np.abs(xp - x))[0]
        if len(changed_elements) > 4:  # this should never happen since we update one tile at a time
            return -100.0
        i = changed_elements[0] / 4
        # since theta (orientation) is periodic, we need to figure out the difference xp-x carefully
        p_i_x = self.get_pick_tile_proposal(x, data, params)
        m_x, sd_x = self.get_move_tile_proposal(x, data, i, params)
        xp_i = xp[i*4:(i+1)*4].copy()
        x_i = x[i*4:(i+1)*4].copy()
        theta_x = x_i[3]
        theta_xp = xp_i[3]
        # we might have gotten to xp from x in two ways (going in the positive or negative direction)
        # therefore, there are two possible actions that get us from x to xp
        # we assume we always do the one with higher probability (i.e., closer to proposal mean)
        dt1 = theta_xp - theta_x
        dt2 = ((2 * X_BOUND) - np.abs(dt1))
        if dt1 > 0.0:
            dt2 = -1.0 * np.abs(dt2)
        else:
            dt2 = np.abs(dt2)
        if np.abs(dt2 - m_x[3]) < np.abs(dt1 - m_x[3]):
            xp_i[3] = dt2
            x_i[3] = 0.0

        q_xp_x = np.log(p_i_x[i])
        q_xp_x += -0.5*(np.sum((xp_i - x_i - m_x)**2 / (sd_x**2))) - 0.5*4*np.log(2*np.pi) - np.sum(np.log(sd_x))
        return q_xp_x

    def propose_probability(self, x, data, xp):
        def _log_ppf(pp):
            return self.log_propose_probability(x, data, xp, pp)

        def _log_ppb(pp):
            return self.log_propose_probability(xp, data, x, pp)

        g_logppf = ag.grad(_log_ppf)
        g_logppb = ag.grad(_log_ppb)
        return np.exp(self.log_propose_probability(x, data, xp, self.params)), g_logppf(self.params), g_logppb(self.params)
