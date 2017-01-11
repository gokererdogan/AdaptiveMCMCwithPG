import numpy as np

import find_the_dot_problem as ftd

run_no = 356653
params = np.load('results/{0:06}_params.npy'.format(run_no))
rewards = np.load('results/{0:06}_rewards.npy'.format(run_no))

my_policy = ftd.FindTheDotPolicy(type='nonlinear', n_hidden=18)
my_policy.params = params[-1]

data = ftd.FindTheDotTargetDistribution.render(np.zeros(1))
for x in ftd.XS:
    m, sd = my_policy.get_proposal_distribution(x, data, params[-1])
    print "position: {0} \t mean, sd: {1}, {2}".format(x, m, sd)

