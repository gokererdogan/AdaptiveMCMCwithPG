"""
Learning data-driven proposals through reinforcement learning

Occluding tiles problem. Taken from
    Jampani, V., Nowozin, S., Loper, M., & Gehler, P. V. (2014). The Informed Sampler: A Discriminative Approach
        to Bayesian Inference in Generative Computer Vision Models. arXiv 1402.0859.

This is a toy visual inference problem used mostly for development and testing purposes. Note that this is a variant
of the problem presented in the above paper.

1 Aug 2016
https://github.com/gokererdogan
"""
import numpy as np
import skimage.draw as draw

from mcmclib.hypothesis import Hypothesis

IMG_SIZE = (50, 50)
MIN_TILE_SIZE = (5, 5)
MAX_TILE_SIZE = (25, 25)


class Tile(object):
    def __init__(self, position=None, size=None, orientation=None):
        self.position = position
        self.size = size
        self.orientation = orientation

        if self.position is None:
            self.position = np.array([np.random.randint(0, IMG_SIZE[0]), np.random.randint(0, IMG_SIZE[1])])

        if self.size is None:
            self.size = np.array([np.random.randint(MIN_TILE_SIZE[0], MAX_TILE_SIZE[0]),
                                  np.random.randint(MIN_TILE_SIZE[1], MAX_TILE_SIZE[1])])

        if self.orientation is None:
            self.orientation = 2 * np.random.rand() * np.pi

    def draw(self, img):
        w, h = self.size
        cx, cy = self.position

        # calculate position of corners by rotating the corners of an upright rectangle.
        x, y = -w, -h  # left top corner
        lt_x = cx + x * np.cos(self.orientation) - y * np.sin(self.orientation)
        lt_y = cy + x * np.sin(self.orientation) + y * np.cos(self.orientation)

        x, y = w, -h  # right top corner
        rt_x = cx + x * np.cos(self.orientation) - y * np.sin(self.orientation)
        rt_y = cy + x * np.sin(self.orientation) + y * np.cos(self.orientation)

        x, y = w, h  # right bottom corner
        rb_x = cx + x * np.cos(self.orientation) - y * np.sin(self.orientation)
        rb_y = cy + x * np.sin(self.orientation) + y * np.cos(self.orientation)

        x, y = -w, h  # left bottom corner
        lb_x = cx + x * np.cos(self.orientation) - y * np.sin(self.orientation)
        lb_y = cy + x * np.sin(self.orientation) + y * np.cos(self.orientation)

        poly_x = np.array([lt_x, rt_x, rb_x, lb_x, lt_x])
        poly_y = np.array([lt_y, rt_y, rb_y, lb_y, lt_y])
        filled_rows, filled_cols = draw.polygon(y=poly_y, x=poly_x, shape=IMG_SIZE)
        img[filled_rows, filled_cols] = 0.5

        border_rows, border_cols = draw.polygon_perimeter(cr=poly_y, cc=poly_x, shape=IMG_SIZE)
        img[border_rows, border_cols] = 1.0

    def copy(self):
        position_copy = self.position.copy()
        size_copy = self.size.copy()
        tile_copy = Tile(position=position_copy, size=size_copy, orientation=self.orientation)
        return tile_copy

    def __str__(self):
        return "pos={0:s}, size={1:s}, orientation={2:f}".format(self.position, self.size, np.degrees(self.orientation))

    def __repr__(self):
        return "pos={0:s}, size={1:s}, orientation={2:f}".format(self.position, self.size, np.degrees(self.orientation))


class OccludingTilesHypothesis(Hypothesis):
    def __init__(self, tiles=None, tile_count=3, ll_variance=0.001):
        Hypothesis.__init__(self)

        self.ll_variance = ll_variance

        self.tiles = tiles
        if self.tiles is None:
            self.tiles = []
            for i in range(tile_count):
                self.tiles.append(Tile())

    def _calculate_log_prior(self):
        return 0.0
        # return len(self.tiles) * np.log(0.5)

    def _calculate_log_likelihood(self, data=None):
        render = self.render()
        return -np.sum(np.square(render - data)) / (2 * self.ll_variance * render.size)

    def render(self):
        img = np.zeros(IMG_SIZE)
        for tile in self.tiles:
            tile.draw(img)
        return img

    def copy(self):
        tiles_copy = []
        for tile in self.tiles:
            tiles_copy.append(tile.copy())
        return OccludingTilesHypothesis(tiles=tiles_copy, ll_variance=self.ll_variance)

    def __str__(self):
        return "\n".join([str(tile) for tile in self.tiles])

    def __repr__(self):
        return "\n".join([repr(tile) for tile in self.tiles])


def add_remove_tile(h, params):
    max_tile_count = np.inf
    if 'MAX_TILE_COUNT' in params.keys():
        max_tile_count = params['MAX_TILE_COUNT']

    hp = h.copy()
    tile_count = len(h.tiles)

    if tile_count > max_tile_count:
        raise ValueError("add/remove tile expects hypothesis with fewer than {0:d} tiles.".format(max_tile_count))

    if tile_count == 0:
        raise ValueError("add/remove tile expects hypothesis to have at least 1 tile.")

    # we cannot add or remove parts if max_tile_count is 1.
    if max_tile_count == 1:
        return hp, 1.0, 1.0

    if tile_count == 1 or (tile_count != max_tile_count and np.random.rand() < .5):
        # add move
        new_tile = Tile()
        hp.tiles.append(new_tile)

        # q(hp|h)
        # NOTE: this is tricky. q(hp|h) is not simply 1/2. After picking the add move, we also need to pick where to add
        # the tile. This may seem unnecessary but it is NOT. Because we care about which tile we remove, we also have to
        # care about where we add the new tile. Therefore, q(hp|h) = 1/2 * (1 / (tile_count + 1))
        q_hp_h = 0.5 * (1.0 / (tile_count + 1))
        # if add is the only move possible
        if tile_count == 1:
            q_hp_h = 1.0 * (1.0 / (tile_count + 1))

        # q(h|hp)
        q_h_hp = 0.5 * (1.0 / (tile_count + 1))
        #  if remove is the only possible reverse move
        if tile_count == (max_tile_count - 1):
            q_h_hp = 1.0 * (1.0 / (tile_count + 1))
    else:
        # remove move
        remove_id = np.random.randint(0, tile_count)
        hp.tiles.pop(remove_id)

        # see the above note in add move
        q_h_hp = 0.5 * (1.0 / tile_count)
        if tile_count == 2:
            q_h_hp = 1.0 * (1.0 / tile_count)

        q_hp_h = 0.5 * (1.0 / tile_count)
        # if remove move is the only possible move
        if tile_count == max_tile_count:
            q_hp_h = 1.0 * (1.0 / tile_count)

    return hp, q_hp_h, q_h_hp


def move_tile(h, params):
    hp = h.copy()
    tile = np.random.choice(hp.tiles)
    tile.position = np.array([np.random.randint(0, IMG_SIZE[0]), np.random.randint(0, IMG_SIZE[1])])
    return hp, 1.0, 1.0


def resize_tile(h, params):
    hp = h.copy()
    tile = np.random.choice(hp.tiles)
    tile.size = np.array([np.random.randint(MIN_TILE_SIZE[0], MAX_TILE_SIZE[0]),
                          np.random.randint(MIN_TILE_SIZE[1], MAX_TILE_SIZE[1])])
    return hp, 1.0, 1.0


def rotate_tile(h, params):
    hp = h.copy()
    tile = np.random.choice(hp.tiles)
    tile.orientation = 2 * np.random.rand() * np.pi
    return hp, 1.0, 1.0


if __name__ == "__main__":
    import mcmclib.proposal

    h = OccludingTilesHypothesis(ll_variance=0.001)

    moves = {'ot_add_remove_tile': add_remove_tile,
             'ot_move_tile': move_tile,
             'ot_resize_tile': resize_tile,
             'ot_rotate_tile': rotate_tile}

    params = {'MAX_TILE_COUNT': 5}

    proposal = mcmclib.proposal.RandomMixtureProposal(moves, params)

    observed = np.load('data/test_ot.npy')

    # choose sampler
    thinning_period = 2000
    sampler_class = 'mh'
    import mcmclib.mh_sampler
    sampler = mcmclib.mh_sampler.MHSampler(h, observed, proposal, burn_in=1000, sample_count=10, best_sample_count=10,
                                           thinning_period=thinning_period, report_period=thinning_period)

    run = sampler.sample()
