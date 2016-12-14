"""
Learning data-driven proposals through reinforcement learning

This file contains the Metropolis-Hastings sampler class.

13 Dec. 2016
https://github.com/gokererdogan
"""
import warnings

import autograd.numpy as np
import autograd as ag


class ParallelMHChains(object):
    """
    ParallelMHChains class implements multiple Metropolis-Hastings chains running
    in parallel. In addition to being a MH sampler, it also returns the derivatives
    of the proposal with respect to its parameters for each timestep.
    """
    def __init__(self, target_distribution, policy, reward_function, chain_count=10, episode_length=40):
        self.target_distribution = target_distribution
        self.policy = policy
        self.reward_function = reward_function
        self.chain_count = chain_count
        self.episode_length = episode_length
        self.x0 = np.array([self.target_distribution.initial_x() for _ in range(self.chain_count)])

    def reset(self):
        self.x0 = np.array([self.target_distribution.initial_x() for _ in range(self.chain_count)])

    def log_acceptance_ratio(self, x, xp, params):
        """
        a(x -> xp)
        """
        logp_x = self.target_distribution.log_probability(x)
        logp_xp = self.target_distribution.log_probability(xp)
        log_q_xp_x = self.policy.log_propose_probability(x, xp, params)
        log_q_x_xp = self.policy.log_propose_probability(xp, x, params)
        log_a = (logp_xp + log_q_x_xp) - (logp_x + log_q_xp_x)
        return log_a

    def acceptance_ratio(self, x, xp):
        def _log_rr(pp):
            return np.log(1.0 - np.exp(self.log_acceptance_ratio(x, xp, pp)))

        g_logrr = ag.grad(_log_rr)
        ar = np.exp(self.log_acceptance_ratio(x, xp, self.policy.params))
        if np.isnan(ar):
            warnings.warn("Acceptance ratio is nan.")
        if ar < 1.0:
            drr = g_logrr(self.policy.params)
        else:
            drr = [np.zeros(p.shape) for p in self.policy.params]

        return ar, drr

    def run_episode(self):
        rewards = np.zeros(self.chain_count)
        xs = np.zeros((self.episode_length,) + self.x0[0].shape)
        accepteds = np.zeros(self.episode_length, dtype=np.bool)
        dlogps = [np.zeros((self.chain_count,) + p.shape) for p in self.policy.params]
        for c in range(self.chain_count):
            x = self.x0[c]
            for t in range(self.episode_length):
                xp = self.policy.propose(x)
                pp, dlog_p_xp_x, dlog_p_x_xp = self.policy.propose_probability(x, xp)
                a, dlog_r = self.acceptance_ratio(x, xp)
                if np.random.rand() < a:  # accept
                    accepteds[t] = True
                    x = xp

                    if a > 1.0:
                        gradient = dlog_p_xp_x
                    else:
                        gradient = dlog_p_x_xp
                else:
                    assert(not type(dlog_r) == float)
                    gradient = [t1+t2 for t1, t2 in zip(dlog_p_xp_x,  dlog_r)]

                # accumulate gradients
                for dl, gi in zip(dlogps, gradient):
                    dl[c] += gi

                # record current sample
                xs[t] = x

            # last sample for this episode will be the first sample in the next
            self.x0[c] = x
            rewards[c] = self.reward_function(self.target_distribution, xs, accepteds)
        return rewards, dlogps

