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
    """
    epoch_count = [5000, 125, 125]
    episode_length = [25, 25, 100]
    episodes_per_epoch = [1, 40, 10]
    save_period = [200, 200, 50]
    report_period = [5, 5, 5]
    plot_period = [100, 100, 25]
    load_from_run = [920863, 156475, 690281]
    """

    epoch_count = [250]
    episode_length = [100]
    episodes_per_epoch = [10]
    save_period = [50]
    report_period = [5]
    plot_period = [25]
    load_from_run = [780732]

    my_target = occluding_tiles_problem.OccludingTilesDistribution(ll_sd=0.02)
    my_policy = [occluding_tiles_problem.OccludingTilesPolicy(learn_pick_tile=False, learn_move_tile=True,
                                                              move_filter_count=50, move_filter_size=(3, 3),
                                                              move_pool_size=(4, 4), move_sd_multiplier=1.0)]
    vimco_estimator = gradient_estimator.VIMCOEstimator(clip=True)
    # learning_rate_schedule = [parameter_schedule.ConstantSchedule(10**(-4.0))]
    learning_rate_schedule = [parameter_schedule.ConstantSchedule(10**(-4.5))]

    """
    reward_type = "Increase in Avg. Log Probability"
    reward_fn = reward.log_prob_increase_avg
    """

    reward_batch_count = 5
    reward_type = "-Autocorrelation"
    def reward_fn(td, x0, xs, accs):
            return reward.auto_correlation_batch_means(td, x0, xs, accs, batch_count=reward_batch_count)

    exp = Experiment(name='occluding_tiles', experiment_method=run_experiment, load_from_run=load_from_run,
                     grouped_params=['epoch_count', 'episode_length', 'episodes_per_epoch', 'save_period',
                                     'report_period', 'plot_period', 'load_from_run'],
                     seed=172014, results_folder='./results/runs', target_distribution=my_target, policy=my_policy,
                     reward_type=reward_type, reward_function=reward_fn, gradient_estimator=vimco_estimator,
                     learning_rate_schedule=learning_rate_schedule, adam_b1=[0.9], adam_b2=[0.999],
                     chain_count=20, episode_length=episode_length, epoch_count=epoch_count,
                     episodes_per_epoch=episodes_per_epoch, save_period=save_period, report_period=report_period,
                     plot_period=plot_period)

    exp.run(parallel=True, num_processes=-1)
    print exp.results
    exp.save('./results/runs/')
    exp.append_csv('./results/runs/occluding_tiles.csv')
