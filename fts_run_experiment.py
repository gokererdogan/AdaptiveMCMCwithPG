"""
Learning Data-Driven Proposals

run_experiment method for running experiments on 2D find the square problem.

https://github.com/gokererdogan
15 Dec. 2016
"""
import numpy as np

import gradient_estimator
import find_the_square_problem
import reward
import parameter_schedule

from run_experiment import run_experiment

from gmllib.experiment import Experiment

if __name__ == "__main__":
    # epoch_count = [500, 500, 500]
    epoch_count = [10000, 5000, 2500]
    # episode_length = [20, 50, 100]
    episode_length = [25, 50, 100]
    # episodes_per_epoch = [25, 10, 5]
    episodes_per_epoch = [1, 1, 1]
    my_target = find_the_square_problem.FindTheSquareTargetDistribution(ll_variance=5.0)
    my_policy = [find_the_square_problem.FindTheSquarePolicy(n_hidden=100)]
    vimco_estimator = gradient_estimator.VIMCOEstimator(clip=True)
    learning_rate_schedule = [parameter_schedule.ConstantSchedule(10**(-3.0))]

    reward_type = "Increase in Avg. Log Probability"
    reward_fn = reward.log_prob_increase_avg

    """
    reward_type = "-Autocorrelation"
    def reward_fn(td, xs, accs):
            return reward.auto_correlation_batch_means(td, xs, accs, batch_count=5)
    """

    exp = Experiment(name='find_the_square', experiment_method=run_experiment,
                     grouped_params=['epoch_count', 'episode_length', 'episodes_per_epoch'],
                     seed=102993, results_folder='./results', target_distribution=my_target, policy=my_policy,
                     reward_type=reward_type, reward_function=reward_fn, gradient_estimator=vimco_estimator,
                     learning_rate_schedule=learning_rate_schedule, adam_b1=[0.9], adam_b2=[0.999],
                     chain_count=20, episode_length=episode_length, epoch_count=epoch_count,
                     episodes_per_epoch=episodes_per_epoch, save_period=100)

    exp.run(parallel=True, num_processes=-1)
    print exp.results
    exp.save('./results')
    exp.append_csv('./results/find_the_square.csv')
