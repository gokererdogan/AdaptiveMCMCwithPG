"""
Learning data-driven proposals through reinforcement learning

Find circles problem. The image consists of a variable number of (fixed size) circles
standing side by side. The task is to find these circles. Here the action space is
discrete where we have one flip (turn on/off circle) action for each possible circle
location.

This is a toy visual inference problem used mostly for development and testing purposes.

2 Aug 2016
https://github.com/gokererdogan
"""
import numpy as np
import skimage.draw as draw

from mcmclib.hypothesis import Hypothesis

IMG_SIZE = (20, 100)
CIRCLE_R = 10
N = 5


class FindCirclesProblem(Hypothesis):
    def __init__(self, configuration=None, ll_variance=0.001):
        Hypothesis.__init__(self)
        self.ll_variance = ll_variance
        self.configuration = configuration
        if self.configuration is None:
            self.configuration = np.random.rand(N) < 0.5

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
        for i, filled in enumerate(self.configuration):
            if filled:
                rows, cols = draw.circle(r=CIRCLE_R, c=(2*i + 1) * CIRCLE_R,
                                         radius=CIRCLE_R, shape=IMG_SIZE)
                image[rows, cols] = 1.0

        return image

    def copy(self):
        config_copy = self.configuration.copy()
        return FindCirclesProblem(configuration=config_copy, ll_variance=self.ll_variance)


def flip_circle(h, params, c_id):
    hp = h.copy()
    hp.configuration[c_id] = ~hp.configuration[c_id]
    return hp, 1.0, 1.0


if __name__ == "__main__":
    import mcmclib.proposal

    h = FindCirclesProblem(ll_variance=.05)

    # note the default parameter i for lambda. This is to get around a problem
    # created by lazy evaluation
    moves = {'fc_flip_circle_{0:d}'.format(i): lambda x, p, i=i: flip_circle(x, p, c_id=i)
             for i in range(N)}

    proposal = mcmclib.proposal.RandomMixtureProposal(moves, params=None)

    observed = np.load('data/test_find_circles.npy')

    # choose sampler
    thinning_period = 500
    sampler_class = 'mh'
    import mcmclib.mh_sampler
    sampler = mcmclib.mh_sampler.MHSampler(h, observed, proposal, burn_in=1000, sample_count=10, best_sample_count=10,
                                           thinning_period=thinning_period, report_period=thinning_period)

    run = sampler.sample()
