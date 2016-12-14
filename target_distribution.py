"""
Learning data-driven proposals through reinforcement learning

This file contains the classes for representing target distributions.

13 Dec. 2016
https://github.com/gokererdogan
"""
import scipy.stats
import autograd.numpy as np


class TargetDistribution(object):
    def __init__(self):
        pass

    def initial_x(self):
        """
        Get initial state

        Returns:
            numpy.ndarray
        """
        raise NotImplementedError()

    def log_probability(self, x):
        raise NotImplementedError()

    def probability(self, x):
        """
        p(x)
        """
        raise NotImplementedError()


class MultivariateGaussian(TargetDistribution):
    def __init__(self, mean, cov):
        """
        Assumes mean and cov are numpy arrays.
        """
        TargetDistribution.__init__(self)
        self.mean = mean
        self.cov = cov
        self.D = self.mean.size

        self.dist = scipy.stats.multivariate_normal(mean=self.mean, cov=self.cov)

    def initial_x(self):
        """
        Get initial state

        Returns:
            numpy.ndarray
        """
        six_sigma = 6 * np.sqrt(np.diag(self.cov))
        return ((np.random.rand(self.D) - 0.5) * 2.0 * six_sigma) + self.mean

    def log_probability(self, x):
        return self.dist.logpdf(x)

    def probability(self, x):
        """
        p(x)
        """
        return self.dist.pdf(x)

