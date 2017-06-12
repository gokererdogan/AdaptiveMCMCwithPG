"""
Learning Data-Driven Proposals

Runs the sampler for 1D Gaussian problem.

https://github.com/gokererdogan
11 Jan. 2016
"""
import numpy as np

import lddp_sampler as sampler
import policy
import target_distribution
import reward
import gradient_estimator
import parameter_schedule
from optimizer import adam

if __name__ == "__main__":
    run_id = 641185
    load_from_i = 50
    params = np.load('results/runs/1d/{0:6d}_params.npy'.format(run_id))
    np.random.seed(1)

    gradient_clip = True
    reward_batch_count = 5

    # my_policy = policy.LinearGaussianPolicy(D=1, mean=np.zeros(1), sd=None)
    my_policy = policy.NonlinearGaussianPolicy(D=1, n_hidden=5, mean=None, sd=None)
    for i, pi in enumerate(params[load_from_i]):
        my_policy.params[i][:] = pi
    my_target = target_distribution.MultivariateGaussian(mean=np.zeros(1), cov=np.eye(1))

    reward_type = "-Autocorrelation"
    def reward_fn(td, x0, xs, accs):
            return reward.auto_correlation_batch_means(td, x0, xs, accs, batch_count=reward_batch_count)

    # my_grad_estimator = gradient_estimator.VIMCOEstimator(clip=gradient_clip)
    my_grad_estimator = gradient_estimator.MeanRewardBaselineEstimator(clip=gradient_clip)
    lr_schedule = parameter_schedule.ConstantSchedule(0.003)

    my_chains = sampler.ParallelMHChains(target_distribution=my_target, policy=my_policy,
                                         reward_function=reward_fn, chain_count=10,
                                         episode_length=100)
    r = my_chains.run_episode()

    """
    params, rewards, grad_magnitudes, acceptance_rates = \
        adam(my_chains, my_grad_estimator,
             learning_rate_schedule=lr_schedule,
             epoch_count=1, episodes_per_epoch=10,
             report_period=1, save_period=1)
    print params
    print rewards
    print grad_magnitudes
    print acceptance_rates
    """
