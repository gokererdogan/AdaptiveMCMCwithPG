"""
Learning Data-Driven Proposals

run_experiment method for running experiments on find the dot problem.

https://github.com/gokererdogan
15 Dec. 2016
"""
import numpy as np

import gradient_estimator
import find_the_dot_problem
import reward
import parameter_schedule

from run_experiment import run_experiment

from gmllib.experiment import Experiment

if __name__ == "__main__":
    epoch_count = [2000, 1000]
    episode_length = [25, 50]
    episodes_per_epoch = [1, 1]
    chain_count = [10, 20]
    my_target = find_the_dot_problem.FindTheDotTargetDistribution(ll_variance=0.2)
    my_policy = [find_the_dot_problem.FindTheDotPolicy(type='linear'),
                 find_the_dot_problem.FindTheDotPolicy(type='nonlinear', n_hidden=18)]

    vimco_estimator = gradient_estimator.VIMCOEstimator(clip=True)
    learning_rate_schedule = [parameter_schedule.ConstantSchedule(10**(-3.0))]
                              # parameter_schedule.ConstantSchedule(10**(-2.0))]

    reward_type = "Increase in Avg. Log Probability"

    exp = Experiment(name='find_the_dot', experiment_method=run_experiment,
                     grouped_params=['epoch_count', 'episode_length', 'episodes_per_epoch'],
                     seed=93102, results_folder='./results', target_distribution=my_target, policy=my_policy,
                     reward_type=reward_type, reward_function=[reward.log_prob_increase_avg],
                     gradient_estimator=vimco_estimator,
                     learning_rate_schedule=learning_rate_schedule, adam_b1=[0.9], adam_b2=[0.999],
                     chain_count=chain_count, episode_length=episode_length, epoch_count=epoch_count,
                     episodes_per_epoch=episodes_per_epoch, save_period=10)

    exp.run(parallel=True, num_processes=-1)
    print exp.results
    exp.save('./results')
    exp.append_csv('./results/find_the_dot.csv')
