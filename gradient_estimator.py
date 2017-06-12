"""
Learning data-driven proposals through reinforcement learning

This file contains the methods for estimating gradients from rewards and derivatives of the proposal.

13 Dec. 2016
https://github.com/gokererdogan
"""
import autograd.numpy as np


class GradientEstimator(object):
    """
    GradientEstimator class implements the estimation of the gradient from rewards and the
    derivatives of the transition probabilities.
    """
    def __init__(self, clip=False):
        self.clip = clip
        pass

    def reset(self):
        pass

    def __str__(self):
        return "GradientEstimator with clip={0}".format(self.clip)

    def __repr__(self):
        return self.__str__()

    def estimate_gradient(self, rewards, grad_logp):
        g = []
        for grad_logp_i in grad_logp:
            gi = np.mean((grad_logp_i.T * rewards).T, axis=0)
            if self.clip:
                if np.sum(np.square(gi)) > 1.0:
                    gi /= np.sum(np.square(gi))
            g.append(gi)
        return g


class MeanRewardBaselineEstimator(GradientEstimator):
    """
    Use the average reward of all chains as baseline.
    """
    def __init__(self, clip=False):
        GradientEstimator.__init__(self, clip=clip)

    def __str__(self):
        return "MeanRewardBaselineEstimator with clip={0}".format(self.clip)

    def __repr__(self):
        return self.__str__()

    def estimate_gradient(self, rewards, grad_logp):
        rewards_baselined = rewards - np.mean(rewards)
        g = GradientEstimator.estimate_gradient(self, rewards_baselined, grad_logp)
        return g


class BBVIEstimator(GradientEstimator):
    """
    Use score function as control variate
    Ranganath, R., Gerrish, S., & Blei, D. M. (2013). Black Box Variational Inference.
    """
    def __init__(self, clip=False):
        GradientEstimator.__init__(self, clip=clip)

    def __str__(self):
        return "BBVIEstimator with clip={0}".format(self.clip)

    def __repr__(self):
        return self.__str__()

    def estimate_gradient(self, rewards, grad_logp):
        cov = 0.0
        var = 0.0
        for grad_logp_i in grad_logp:
            dl = np.reshape(grad_logp_i, (grad_logp_i.shape[0], -1))
            dim = dl.shape[1]
            for i in range(dim):
                f = rewards * dl[:, i]
                g = dl[:, i]  # control variate
                cov_mat = np.cov(f, g)
                cov += cov_mat[0, 1]  # Cov(f, g)
                var += cov_mat[1, 1]  # Var(g)
        # calculate optimal scaling factor
        a_opt = cov / var

        rewards_baselined = rewards - a_opt
        g = GradientEstimator.estimate_gradient(self, rewards_baselined, grad_logp)
        return g


class VIMCOEstimator(GradientEstimator):
    """
    Use the average reward of the other chains as baseline.
    Inspired by Mnih and Rezende, Variational inference for Monte Carlo objectives
    """
    def __init__(self, clip=False):
        GradientEstimator.__init__(self, clip=clip)

    def __str__(self):
        return "VIMCOEstimator with clip={0}".format(self.clip)

    def __repr__(self):
        return self.__str__()

    def estimate_gradient(self, rewards, grad_logp):
        K = rewards.shape[0]
        tot_r = np.sum(rewards)
        # rewards_baselined = (tot_r / K) - ((tot_r - rewards) / (K-1))
        rewards_baselined = rewards - ((tot_r - rewards) / (K-1))
        g = GradientEstimator.estimate_gradient(self, rewards_baselined, grad_logp)
        return g

