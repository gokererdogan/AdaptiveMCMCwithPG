"""
Learning data-driven proposals through reinforcement learning

Occluding tiles problem. Taken from
    Jampani, V., Nowozin, S., Loper, M., & Gehler, P. V. (2014). The Informed Sampler: A Discriminative Approach
        to Bayesian Inference in Generative Computer Vision Models. arXiv 1402.0859.

9 Jan. 2016
https://github.com/gokererdogan
"""
from collections import OrderedDict

import skimage.draw as draw

import autograd.numpy as np
from autograd.scipy.signal import convolve

from target_distribution import TargetDistribution
from policy import Policy

MIN_TILE_SIZE = (10, 10)
MAX_TILE_SIZE = (30, 30)
IMG_SIZE = (50, 50)
X_BOUND = 6.0
COLORS = [(0.969, 0.076, 0.076), (0.995, 0.995, 0.161), (0.141, 0.978, 0.141),
          (0.165, 0.960, 0.960), (0.076, 0.076, 0.988), (0.981, 0.162, 0.981)]
OUT_OF_BOUNDS_PENALTY = 10e6


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

        img = OccludingTilesDistribution.render_rgb(x)

        mimg = np.zeros((IMG_SIZE + (6,)))
        for ci, c in enumerate(COLORS):
            mimg[np.all(np.abs(img - c) < 1e-3, axis=2), ci] = 1.0

        render_cache.add(x, mimg)
        return mimg

    @staticmethod
    def render_rgb(x):
        img = np.zeros(IMG_SIZE + (3,))
        # sort tiles according to depth. start from the tile that is farthest away.
        zs = x[2::4]  # get every 3rd element
        ix = np.argsort(zs)
        for i in ix:
            OccludingTilesDistribution.draw_tile(x[i*4:(i+1)*4], img, COLORS[i])
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
    def __init__(self, learn_pick_tile=True, learn_move_tile=True, pick_filter_count=20, move_filter_count=50,
                 move_filter_size=(3, 3), move_pool_size=(4, 4), move_sd_multiplier=1.0):
        Policy.__init__(self)
        self.learn_pick_tile = learn_pick_tile
        self.learn_move_tile = learn_move_tile
        self.move_sd_multiplier = move_sd_multiplier

        self.params = []
        if self.learn_pick_tile:
            self.pick_filter_count = pick_filter_count
            # pick tile network params
            self.params.append(np.random.randn(6, self.pick_filter_count) * 0.01)
            self.params.append(np.random.randn(self.pick_filter_count) * 0.01)
            self.params.append(np.random.randn(self.pick_filter_count, 6) * 0.01)
            self.params.append(np.random.randn(6) * 0.001)

        if self.learn_move_tile:
            self.move_filter_count = move_filter_count
            self.move_filter_size = move_filter_size
            self.move_pool_size = move_pool_size
            filtered_img_size = (IMG_SIZE[0] - self.move_filter_size[0] + 1, IMG_SIZE[1] - self.move_filter_size[1] + 1)
            assert (filtered_img_size[0] % self.move_pool_size[0] == 0 and
                    filtered_img_size[1] % self.move_pool_size[1] == 0), "Filter size and pool size are incompatible."
            # move tile network params
            self.params.append(np.random.randn(self.move_filter_count, *self.move_filter_size) * 0.01)
            self.params.append(np.random.randn(self.move_filter_count) * 0.01)
            pooled_img_size = self.move_filter_count * (filtered_img_size[0] // self.move_pool_size[0]) * \
                              (filtered_img_size[1] // self.move_pool_size[1])
            self.params.append(np.random.randn(pooled_img_size, 4) * 0.01)
            self.params.append(np.random.randn(4) * 0.01)

    def __str__(self):
        s = "OccludingTilesPolicy with sd_mult={0}".format(self.move_sd_multiplier)
        if self.learn_pick_tile:
            s += "; learned pick_tile (filter_count={0})".format(self.pick_filter_count)
        if self.learn_move_tile:
            s += "; learned move_tile (filter_count={0}, " \
                 "filter_size={1})".format(self.move_filter_count, self.move_filter_size)
        return s

    def __repr__(self):
        return self.__str__()

    def get_pick_tile_proposal(self, x, data, params):
        if not self.learn_pick_tile:
            return np.ones(6) / 6.0

        img = OccludingTilesDistribution.render(x)
        nn_input = img - data

        # pick tile network
        filter_w = params[0]
        filter_b = params[1]
        out_w = params[2]
        out_b = params[3]
        # conv + relu
        filtered_input = np.maximum(np.dot(nn_input, filter_w) + filter_b, 0)
        # global pooling (take mean of each channel) + fully connected
        pick_tile_outputs = np.dot(np.mean(filtered_input, axis=(0, 1)), out_w) + out_b
        pick_tile_probs = np.exp(pick_tile_outputs) / np.sum(np.exp(pick_tile_outputs))

        return pick_tile_probs

    def get_move_tile_proposal(self, x, data, tile, params):
        if not self.learn_move_tile:
            return np.zeros(4), self.move_sd_multiplier * np.ones(4)

        # pick tile network
        offset = 0
        if self.learn_pick_tile:
            offset = 4
        filter_w = params[offset + 0]
        filter_b = params[offset + 1]
        out_w = params[offset + 2]
        out_b = params[offset + 3]

        # form input
        img = OccludingTilesDistribution.render(x)
        nn_input = np.reshape((img[:, :, tile] - data[:, :, tile]), (1, 1, IMG_SIZE[0], IMG_SIZE[1]))

        # conv + relu
        filter_w = np.reshape(filter_w, (1,) + filter_w.shape)
        filtered_input = np.maximum(convolve(nn_input, filter_w, axes=([2, 3], [2, 3]), dot_axes=([1], [0]), mode='valid') + \
                                    np.reshape(filter_b, (-1, 1, 1)), 0)[0]
        # max pooling
        pooled_input = np.max(np.max(np.reshape(filtered_input, (filtered_input.shape[0],
                                                                 filtered_input.shape[1]//self.move_pool_size[0],
                                                                 self.move_pool_size[0],
                                                                 filtered_input.shape[2]//self.move_pool_size[1],
                                                                 self.move_pool_size[1])),
                              axis=2), axis=3)

        # fc
        mean = np.dot(np.ravel(pooled_input), out_w) + out_b
        sd = self.move_sd_multiplier * np.ones(4)

        return mean, sd

    def propose(self, x, data):
        # pick tile
        p_i = self.get_pick_tile_proposal(x, data, self.params)
        i = np.random.choice(6, p=p_i)

        # move tile
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
        changed_tile = changed_elements[0] / 4

        # since theta (orientation) is periodic, we need to figure out the difference xp-x carefully
        p_i_x = self.get_pick_tile_proposal(x, data, params)
        m_x, sd_x = self.get_move_tile_proposal(x, data, changed_tile, params)
        xp_i = xp[changed_tile*4:(changed_tile+1)*4].copy()
        x_i = x[changed_tile*4:(changed_tile+1)*4].copy()
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

        q_xp_x = np.log(p_i_x[changed_tile])
        q_xp_x += -0.5*(np.sum((xp_i - x_i - m_x)**2 / (sd_x**2))) - 0.5*4*np.log(2*np.pi) - np.sum(np.log(sd_x))
        return q_xp_x


class OccludingTilesPolicyFull(Policy):
    """
    Learns a one-stage policy that maps from image directly to a move for all tiles.
    """
    def __init__(self, n_hidden, sd_fixed=True, sd_multiplier=1.0):
        Policy.__init__(self)
        assert n_hidden >= 0
        self.n_hidden = n_hidden
        if self.n_hidden == 0:
            # no hidden layer
            self.has_hidden_layer = False
        else:
            self.has_hidden_layer = True

        self.sd_fixed = sd_fixed
        self.sd_multiplier = sd_multiplier

        self.params = []
        if self.has_hidden_layer:
            self.params.append(np.random.randn(IMG_SIZE[0]*IMG_SIZE[1]*6, self.n_hidden) * 0.001)
            self.params.append(np.random.randn(self.n_hidden) * 0.001)
            if self.sd_fixed:
                self.params.append(np.random.randn(self.n_hidden, 24) * 0.001)
            else:
                self.params.append(np.random.randn(self.n_hidden, 48) * 0.001)
        else:
            if self.sd_fixed:
                self.params.append(np.random.randn(IMG_SIZE[0]*IMG_SIZE[1]*6, 24) * 0.001)
            else:
                self.params.append(np.random.randn(IMG_SIZE[0]*IMG_SIZE[1]*6, 48) * 0.001)

    def __str__(self):
        s = "OccludingTilesPolicyFull with n_hidden={0}, sd_multiplier={1}, sd learned? {2}".format(self.n_hidden,
                                                                                                    self.sd_multiplier,
                                                                                                    self.sd_fixed)
        return s

    def __repr__(self):
        return self.__str__()

    def get_proposal_distribution(self, x, data, params):
        img = OccludingTilesDistribution.render(x)
        nn_input = np.ravel(img - data)

        if self.has_hidden_layer:
            w_hidden = params[0]
            b_hidden = params[1]
            hidden_activations = np.maximum(np.dot(nn_input, w_hidden) + b_hidden, 0.0)
            nn_input = hidden_activations

        w_output = params[-1]
        out = np.dot(nn_input, w_output)
        mean = out[0:24]
        if self.sd_fixed:
            sd = np.ones(24) * self.sd_multiplier
        else:
            sd = np.exp(out[24:]) * self.sd_multiplier
        return mean, sd

    def propose(self, x, data):
        # move tiles
        mean, sd = self.get_proposal_distribution(x, data, self.params)
        a = mean + sd*np.random.randn(mean.size)
        xp = x.copy()
        xp += a
        # theta (orientation) is periodic. It wraps around -pi/4 when it exceeds pi/4 and vice versa
        xp[3::4] = ((xp[3::4] + X_BOUND) % (2.0 * X_BOUND)) - X_BOUND

        return xp

    def log_propose_probability(self, x, data, xp, params):
        m_x, sd_x = self.get_proposal_distribution(x, data, params)
        q_xp_x = -0.5*(np.sum((xp - x - m_x)**2 / (sd_x**2))) - 0.5*m_x.shape[0]*np.log(2*np.pi) - np.sum(np.log(sd_x))
        return q_xp_x
