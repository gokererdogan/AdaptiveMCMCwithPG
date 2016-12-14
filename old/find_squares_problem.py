"""
Learning data-driven proposals through reinforcement learning

Find squares problem. The image consists of a variable number of (fixed size) squares
arranged in a grid. The task is to find these squares. Here the action space is
discrete where we have one flip (turn on/off circle) action for each possible square
location.

This is a toy visual inference problem used mostly for development and testing purposes.

2 Aug 2016
https://github.com/gokererdogan
"""
import numpy as np

from mcmclib.hypothesis import Hypothesis

IMG_SIZE = (40, 40)
SQUARES_PER_SIDE = 4
SQUARE_WIDTH = IMG_SIZE[0] / SQUARES_PER_SIDE
SQUARE_HEIGHT = IMG_SIZE[1] / SQUARES_PER_SIDE


class FindSquaresProblem(Hypothesis):
    def __init__(self, configuration=None, ll_variance=0.001):
        Hypothesis.__init__(self)
        self.ll_variance = ll_variance
        self.configuration = configuration
        if self.configuration is None:
            self.configuration = np.random.rand(SQUARES_PER_SIDE, SQUARES_PER_SIDE) < 0.5

    def _calculate_log_prior(self):
        return 0.0

    def _calculate_log_likelihood(self, data=None):
        render = self.render()
        return -np.sum(np.square(render - data)) / (2 * self.ll_variance * render.size)

    def __str__(self):
        return str(self.configuration)

    def __reduce__(self):
        # for pickle
        return repr(self.configuration)

    def render(self):
        image = np.zeros(IMG_SIZE)
        for i in range(SQUARES_PER_SIDE):
            for j in range(SQUARES_PER_SIDE):
                filled = self.configuration[i, j]
                if filled:
                    image[(i*SQUARE_WIDTH):((i+1)*SQUARE_WIDTH), (j*SQUARE_HEIGHT):((j+1)*SQUARE_HEIGHT)] = 1.0

        return image

    def copy(self):
        config_copy = self.configuration.copy()
        return FindSquaresProblem(configuration=config_copy, ll_variance=self.ll_variance)


def flip_square(h, square_index):
    hp = h.copy()
    hp.configuration[square_index] = ~hp.configuration[square_index]
    return hp


def random_flip_move(h, params):
    square_index = np.random.randint(SQUARES_PER_SIDE), np.random.randint(SQUARES_PER_SIDE)
    hp = flip_square(h, square_index)
    return hp, 1.0, 1.0


if __name__ == "__main__":
    import mcmclib.proposal

    moves = {"fc_random_flip": random_flip_move}

    proposal = mcmclib.proposal.RandomMixtureProposal(moves, params=None)

    import mcmclib.mh_sampler

    likelihood_variance = 0.01
    run_count = 20
    sample_count = 5
    thinning_period = 50
    sampler_class = 'mh'
    runs = []
    avg_log_prob = np.zeros(sample_count*thinning_period)
    for r in range(run_count):
        h = FindSquaresProblem(ll_variance=likelihood_variance)
        observed = FindSquaresProblem(ll_variance=likelihood_variance).render()
        sampler = mcmclib.mh_sampler.MHSampler(h, observed, proposal, burn_in=0, sample_count=sample_count,
                                               best_sample_count=sample_count, thinning_period=thinning_period,
                                               report_period=thinning_period)
        run = sampler.sample()
        runs.append(run)
        avg_log_prob += run.run_log.LogProbability

    avg_log_prob /= run_count


