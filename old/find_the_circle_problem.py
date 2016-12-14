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
CIRCLE_R = 6


class FindTheCircleProblem(Hypothesis):
    def __init__(self, position=None, ll_variance=0.001):
        Hypothesis.__init__(self)
        self.ll_variance = ll_variance
        self.position = position
        if self.position is None:
            self.position = (np.random.rand(2) * IMG_SIZE)

    def _calculate_log_prior(self):
        if 0.0 < self.position[0] < IMG_SIZE[0] and 0.0 < self.position[1] < IMG_SIZE[1]:
            return 0.0
        else:
            # if circle is out of bounds, return a very low prior probability.
            return -100.0

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
        rows, cols = draw.circle(r=int(self.position[0]), c=int(self.position[1]), radius=CIRCLE_R*1.8, shape=IMG_SIZE)
        image[rows, cols] = 0.05
        rows, cols = draw.circle(r=int(self.position[0]), c=int(self.position[1]), radius=CIRCLE_R*1.6, shape=IMG_SIZE)
        image[rows, cols] = 0.35
        rows, cols = draw.circle(r=int(self.position[0]), c=int(self.position[1]), radius=CIRCLE_R*1.4, shape=IMG_SIZE)
        image[rows, cols] = 0.7
        rows, cols = draw.circle(r=int(self.position[0]), c=int(self.position[1]), radius=CIRCLE_R*1.2, shape=IMG_SIZE)
        image[rows, cols] = 0.9
        rows, cols = draw.circle(r=int(self.position[0]), c=int(self.position[1]), radius=CIRCLE_R, shape=IMG_SIZE)
        image[rows, cols] = 1.0

        return image

    def copy(self):
        pos_copy = self.position.copy()
        return FindTheCircleProblem(position=pos_copy, ll_variance=self.ll_variance)


def move_circle(h, move):
    hp = h.copy()
    hp.position += move
    return hp


def random_move_circle_move(h, params):
    # sample move
    move = np.random.randn(2) * params['MOVE_STD_DEV']
    hp = move_circle(h, move)
    return hp, 1.0, 1.0


if __name__ == "__main__":
    import mcmclib.proposal

    moves = {'random_move_circle': random_move_circle_move}

    proposal = mcmclib.proposal.RandomMixtureProposal(moves, params={'MOVE_STD_DEV': 10.0})

    import mcmclib.mh_sampler

    likelihood_variance = 0.0005
    run_count = 20
    sample_count = 5
    thinning_period = 50
    sampler_class = 'mh'
    runs = []
    avg_log_prob = np.zeros(sample_count*thinning_period)
    for r in range(run_count):
        h = FindTheCircleProblem(ll_variance=likelihood_variance)
        observed = FindTheCircleProblem(ll_variance=likelihood_variance).render()
        sampler = mcmclib.mh_sampler.MHSampler(h, observed, proposal, burn_in=0, sample_count=sample_count,
                                               best_sample_count=sample_count, thinning_period=thinning_period,
                                               report_period=thinning_period)
        run = sampler.sample()
        runs.append(run)
        avg_log_prob += run.run_log.LogProbability

    avg_log_prob /= run_count
