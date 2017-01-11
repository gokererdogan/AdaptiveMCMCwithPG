"""
Learning Data-Driven Proposals

run_experiment method for running experiments on 2D occluding tiles problem.

https://github.com/gokererdogan
10 Jan. 2016
"""
import numpy as np

import gradient_estimator
import occluding_tiles_problem
import reward
import parameter_schedule

from run_experiment import run_experiment

from gmllib.experiment import Experiment

if __name__ == "__main__":
    epoch_count = [2000, 2000, 2000]
    episode_length = [100, 50, 25]
    episodes_per_epoch = [10, 20, 40]
    my_target = occluding_tiles_problem.OccludingTilesDistribution(ll_sd=0.02)
    my_policy = [occluding_tiles_problem.OccludingTilesPolicy(n_hidden=100)]
    vimco_estimator = gradient_estimator.VIMCOEstimator(clip=True)
    learning_rate_schedule = [parameter_schedule.ConstantSchedule(10**(-3.0))]

    reward_type = "Increase in Avg. Log Probability"
    reward_fn = reward.log_prob_increase_avg

    exp = Experiment(name='occluding_tiles', experiment_method=run_experiment,
                     grouped_params=['epoch_count', 'episode_length', 'episodes_per_epoch'],
                     seed=391738, results_folder='./results', target_distribution=my_target, policy=my_policy,
                     reward_type=reward_type, reward_function=reward_fn, gradient_estimator=vimco_estimator,
                     learning_rate_schedule=learning_rate_schedule, adam_b1=[0.9], adam_b2=[0.999],
                     chain_count=20, episode_length=episode_length, epoch_count=epoch_count,
                     episodes_per_epoch=episodes_per_epoch, save_period=100)

    exp.run(parallel=True, num_processes=-1)
    print exp.results
    exp.save('./results')
    exp.append_csv('./results/occluding_tiles.csv')
