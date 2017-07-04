"""
Learning Data-Driven Proposals

Runs the experiments for learning a data-driven proposals for sampling from a
1D Gaussian.

https://github.com/gokererdogan
6 Dec. 2016
"""
import policy
import target_distribution
import reward
import gradient_estimator
import parameter_schedule
import optimizer
from run_experiment import *

from gmllib.experiment import Experiment

if __name__ == "__main__":
    seed = 1431
    gradient_clip = True
    reward_batch_count = 5

    episode_length = 100
    episodes_per_epoch = 10
    # epoch_count = 250
    epoch_count = 200
    save_period = 40
    plot_period = 20
    report_period = 1

    # my_policies = policy.LinearGaussianPolicy(D=1, mean=None, sd=np.ones(1))
    my_policies = policy.NonlinearGaussianPolicy(D=1, n_hidden=5, mean=None, sd=None, seed=seed)
    # my_policies = policy.LinearGaussianPolicy(D=1, mean=None, sd=None)
    # my_policies = [policy.LinearGaussianPolicy(D=1, mean=None, sd=np.ones(1)),
    #                policy.LinearGaussianPolicy(D=1, mean=np.zeros(1), sd=None)]
    my_target = target_distribution.MultivariateGaussian(mean=np.zeros(1), cov=np.eye(1))

    # optim_fn = [optimizer.adam, optimizer.sga]
    optim_fn = optimizer.adam
    reward_type = "-Autocorrelation"
    def reward_fn(td, x0, xs, accs):
            return reward.auto_correlation_batch_means(td, x0, xs, accs, batch_count=reward_batch_count)

    """
    reward_type = "Log Probability"
    def reward_fn(td, xs, accs):
            return reward.log_prob(td, xs, accs)
    """

    # my_grad_estimator = gradient_estimator.VIMCOEstimator(clip=gradient_clip)
    my_grad_estimator = gradient_estimator.MeanRewardBaselineEstimator(clip=gradient_clip)

    # lr_schedule = parameter_schedule.LinearSchedule(start_value=0.01, end_value=0.0, decrease_for=50*35)
    lr_schedule = parameter_schedule.ConstantSchedule(0.003)

    exp = Experiment(name='lddp_1D_Gaussian', experiment_method=run_experiment,
                     seed=seed, results_folder='./results/runs/1d', target_distribution=my_target,
                     policy=my_policies, load_from_run=None, optimizer=optim_fn,
                     reward_type=reward_type, reward_function=reward_fn,
                     gradient_estimator=my_grad_estimator,
                     learning_rate_schedule=lr_schedule,
                     chain_count=20, episode_length=episode_length, epoch_count=epoch_count,
                     episodes_per_epoch=episodes_per_epoch,
                     save_period=save_period, plot_period=plot_period, report_period=report_period)

    exp.run(parallel=False, num_processes=-1)
    print exp.results
    exp.save('./results/runs/1d')
    exp.append_csv('./results/runs/1d/results_1d.csv')

