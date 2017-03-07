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
    def __init__(self, target_distribution, policy, reward_function, chain_count=10, episode_length=40,
                 thinning_period=20):
        self.target_distribution = target_distribution
        self.policy = policy
        self.reward_function = reward_function
        self.chain_count = chain_count
        self.episode_length = episode_length
        self.thinning_period = thinning_period
        self.samples_per_episode = int(np.floor(self.episode_length / self.thinning_period))
        self.x0 = np.array([self.target_distribution.initial_x() for _ in range(self.chain_count)])

        # get function for gradient of reject case
        # (the gradients for the accept case do not involve the gradient of acceptance ratio)
        self.grad_log_reject = ag.grad(self._log_rr, 3)

    def reset(self):
        # let the target distribution know we are starting a new chain.
        self.target_distribution.reset()
        self.x0 = np.array([self.target_distribution.initial_x() for _ in range(self.chain_count)])

    def log_acceptance_ratio(self, x, data, xp, params):
        """
        a(x -> xp)
        """
        logp_x = self.target_distribution.log_probability(x)
        logp_xp = self.target_distribution.log_probability(xp)
        log_q_xp_x = self.policy.log_propose_probability(x, data, xp, params)
        log_q_x_xp = self.policy.log_propose_probability(xp, data, x, params)
        log_a = (logp_xp + log_q_x_xp) - (logp_x + log_q_xp_x)
        return log_a

    def _log_rr(self, x, data, xp, params):
        return np.log(1.0 - np.exp(self.log_acceptance_ratio(x, data, xp, params)))

    def acceptance_ratio(self, x, data, xp):
        ar = np.exp(self.log_acceptance_ratio(x, data, xp, self.policy.params))
        if np.isnan(ar):
            warnings.warn("Acceptance ratio is nan.")

        return ar

    def run_episode(self):
        data = self.target_distribution.data
        rewards = np.zeros(self.chain_count)
        acceptance_rates = np.zeros(self.chain_count)
        samples = np.zeros((self.chain_count, self.samples_per_episode) + self.x0[0].shape)
        xs = np.zeros((self.episode_length,) + self.x0[0].shape)
        accepteds = np.zeros(self.episode_length, dtype=np.bool)
        dlogps = [np.zeros((self.chain_count,) + p.shape) for p in self.policy.params]
        for c in range(self.chain_count):
            x = self.x0[c]
            for t in range(self.episode_length):
                xp = self.policy.propose(x, data)
                a = self.acceptance_ratio(x, data, xp)
                if np.random.rand() < a:  # accept
                    # calculate gradient
                    if a > 1.0:
                        # gradient is dlog p(xp|x) / dparams
                        gradient = self.policy.grad_log_propose_probability(x, data, xp, self.policy.params)
                    else:
                        # gradient is dlog p(x|xp) / dparams
                        gradient = self.policy.grad_log_propose_probability(xp, data, x, self.policy.params)

                    accepteds[t] = True
                    x = xp
                else:
                    # calculate gradient of reject probability
                    drr = self.grad_log_reject(x, data, xp, self.policy.params)
                    dlog_p_xp_x = self.policy.grad_log_propose_probability(x, data, xp, self.policy.params)
                    gradient = [t1+t2 for t1, t2 in zip(dlog_p_xp_x,  drr)]

                    accepteds[t] = False

                # accumulate gradients
                for dl, gi in zip(dlogps, gradient):
                    dl[c] += gi

                # record current sample
                xs[t] = x

                # add current sample to samples list
                if (t+1) % self.thinning_period == 0:
                    sample_id = int(np.floor((t+1) / self.thinning_period)) - 1
                    samples[c, sample_id] = x

            acceptance_rates[c] = np.mean(accepteds)
            rewards[c] = self.reward_function(self.target_distribution, self.x0[c], xs, accepteds)
            # last sample for this episode will be the first sample in the next
            self.x0[c] = x
        return rewards, dlogps, samples, acceptance_rates

