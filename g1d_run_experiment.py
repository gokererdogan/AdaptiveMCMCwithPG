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
from run_experiment import *

from gmllib.experiment import Experiment

if __name__ == "__main__":
    gradient_clip = True
    reward_batch_count = 5

    my_policies = policy.LinearGaussianPolicy(D=1, mean=None, sd=None)
    # my_policies = [policy.LinearGaussianPolicy(D=1, mean=None, sd=np.ones(1)),
    #                policy.LinearGaussianPolicy(D=1, mean=np.zeros(1), sd=None)]
    my_target = target_distribution.MultivariateGaussian(mean=np.zeros(1), cov=np.eye(1))

    reward_type = "-Autocorrelation"
    def reward_fn(td, x0, xs, accs):
            return reward.auto_correlation_batch_means(td, x0, xs, accs, batch_count=reward_batch_count)

    """
    reward_type = "Log Probability"
    def reward_fn(td, xs, accs):
            return reward.log_prob(td, xs, accs)
    """

    my_grad_estimator = gradient_estimator.VIMCOEstimator(clip=gradient_clip)
    # lr_schedule = parameter_schedule.LinearSchedule(start_value=0.01, end_value=0.0, decrease_for=50*35)
    lr_schedule = parameter_schedule.ConstantSchedule(0.003)

    exp = Experiment(name='lddp_1D_Gaussian', experiment_method=run_experiment,
                     seed=[345], results_folder='./results', target_distribution=my_target,
                     policy=my_policies,
                     reward_type=reward_type, reward_function=reward_fn,
                     gradient_estimator=my_grad_estimator,
                     learning_rate_schedule=lr_schedule, adam_b1=[0.9], adam_b2=[0.999],
                     chain_count=20, episode_length=100, epoch_count=250, episodes_per_epoch=10,
                     save_period=1)

    exp.run(parallel=True, num_processes=-1)
    print exp.results
    exp.save('./results')
    exp.append_csv('./results/results_1d.csv')

