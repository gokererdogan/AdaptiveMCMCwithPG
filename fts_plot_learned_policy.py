import numpy as np

import find_the_square_problem as fts

run_no = 19804
params = np.load('results/{0:06}_params.npy'.format(run_no))
rewards = np.load('results/{0:06}_rewards.npy'.format(run_no))

my_policy = fts.FindTheSquarePolicy(n_hidden=100)
my_policy.params = params[-1]

xx, yy = np.meshgrid(fts.XS, fts.YS)
means = np.zeros(xx.shape + (2,))
sds = np.zeros(xx.shape + (2,))
data = fts.FindTheSquareTargetDistribution.render(np.array([3.0, 3.0]))
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        x = xx[i, j]
        y = yy[i, j]
        m, sd = my_policy.get_proposal_distribution(np.array([x, y]), data, params[-1])
        means[i, j] = m
        sds[i, j] = sd
        # print "position: {0}, {1} \t mean, sd: {2}, {3}".format(x, y, m, sd)

import matplotlib.pyplot as plt
plt.quiver(xx, yy, means[:, :, 0], means[:, :, 1])
plt.show()
