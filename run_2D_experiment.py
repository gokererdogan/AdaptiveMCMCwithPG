"""
Learning Data-Driven Proposals

Runs the experiments for learning a data-driven proposals for sampling from a
2D Gaussian.

https://github.com/gokererdogan
8 Dec. 2016
"""
from run_experiment import *
from gmllib.experiment import Experiment

if __name__ == "__main__":
    lr_schedule = parameter_schedule.LinearSchedule(start_value=0.01, end_value=0.0, decrease_for=50*35)
    exp = Experiment(name='lddp_2D_Gaussian', experiment_method=run_experiment,
                     seed=1, results_folder='./results', policy_type=['mean+sd', 'mean'], target_dimensionality=2,
                     target_cov=np.array([[1.0, 0.5], [0.5, 1.0]]), policy_linearity='nonlinear',
                     hidden_count=5, reward_type=['-Autocorrelation'], gradient_estimator=['vimco'],
                     learning_rate=lr_schedule, adam_b1=[0.9], adam_b2=[0.999], gradient_clip=True,
                     chain_count=10, episode_length=100, epoch_count=40,
                     episodes_per_epoch=50, batch_count=5)

    exp.run(parallel=True, num_processes=-1)
    print exp.results
    exp.save('./results')
    exp.append_csv('./results/results_2d.csv')

