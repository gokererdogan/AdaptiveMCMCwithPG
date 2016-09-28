"""
Learning data-driven proposals through reinforcement learning

Find the circle problem. The image contains a single circle. The task is to find where this circle is.
Here the action space is continuous because an action specifies where to move the circle center in 2D
image.

This is a toy visual inference problem used mostly for development and testing purposes.

21 September 2016
https://github.com/gokererdogan
"""
import numpy as np
import skimage.draw as draw

from mcmclib.hypothesis import Hypothesis

IMG_SIZE = (50, 50)
CIRCLE_R = 8


class FindTheCircleProblem(Hypothesis):
    def __init__(self, position=None, ll_variance=0.001):
        Hypothesis.__init__(self)
        self.ll_variance = ll_variance
        self.position = position
        if self.position is None:
            self.position = (np.random.rand(2) * IMG_SIZE)

    def _calculate_log_prior(self):
        return 0.0

    def _calculate_log_likelihood(self, data=None):
        render = self.render()
        return -np.sum(np.square(render - data)) / (2 * self.ll_variance * render.size)

    def __str__(self):
        return str(self.position)

    def __repr__(self):
        return repr(self.position)

    def __eq__(self, other):
        if np.sum(np.abs(self.position - other.position)) < 2.0:
            return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def render(self):
        image = np.zeros(IMG_SIZE)
        rows, cols = draw.circle(r=int(self.position[0]), c=int(self.position[1]), radius=CIRCLE_R, shape=IMG_SIZE)
        image[rows, cols] = 1.0
        return image

    def copy(self):
        pos_copy = self.position.copy()
        return FindTheCircleProblem(position=pos_copy, ll_variance=self.ll_variance)


def move_circle(h, params, move=None):
    if move is None:
        # sample move
        move = np.random.randn(2) * params['MOVE_STD_DEV']

    new_pos = h.position + move
    if np.any(np.logical_or(new_pos < 0, new_pos > IMG_SIZE)):  # out of bounds
        return h, 1.0, 1.0

    hp = h.copy()
    hp.position += move
    return hp, 1.0, 1.0


if __name__ == "__main__":
    import mcmclib.proposal

    h = FindTheCircleProblem(ll_variance=.001)

    moves = {'move_circle': move_circle}

    proposal = mcmclib.proposal.RandomMixtureProposal(moves, params={'MOVE_STD_DEV': 10.0})

    observed = np.load('data/test_find_the_circle.npy')

    # choose sampler
    thinning_period = 500
    sampler_class = 'mh'
    import mcmclib.mh_sampler
    sampler = mcmclib.mh_sampler.MHSampler(h, observed, proposal, burn_in=1000, sample_count=10, best_sample_count=10,
                                           thinning_period=thinning_period, report_period=thinning_period)

    run = sampler.sample()
