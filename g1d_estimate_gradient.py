import numpy as np

import target_distribution
import policy

import lddp_sampler
import reward
import gradient_estimator


if __name__ == "__main__":
    np.random.seed(1)

    episode_length = 100
    my_target = target_distribution.MultivariateGaussian(mean=np.zeros(1), cov=np.eye(1))

    reward_type = "-Autocorrelation"
    def reward_fn(td, x0, xs, accs):
            return reward.auto_correlation_batch_means(td, x0, xs, accs, batch_count=5)
    """
    reward_fn = reward.log_prob_increase_avg
    """

    vimco = gradient_estimator.VIMCOEstimator(clip=False)
    bbvi = gradient_estimator.BBVIEstimator(clip=False)

    my_policy = policy.LinearGaussianPolicy(D=1, mean=np.zeros(1), sd=None)
    old_params = []
    for p in my_policy.params:
        p[:] = 0.01*np.random.randn(*p.shape)
        old_params.append(p.copy())

    chain_count = 250
    seed = 91844
    print(1)
    my_chains = lddp_sampler.ParallelMHChains(target_distribution=my_target, policy=my_policy,
                                              reward_function=reward_fn,
                                              chain_count=chain_count,
                                              episode_length=episode_length, thinning_period=10)

    np.random.seed(seed)

    rewards, dlogps, samples, acceptance_rates = my_chains.run_episode()
    grad = vimco.estimate_gradient(rewards, dlogps)
    grad_bbvi = bbvi.estimate_gradient(rewards, dlogps)

    lr = 0.01
    for p, g in zip(my_policy.params, grad):
        p += lr * g

    # estimate reward with new params
    print(2)
    np.random.seed(seed)
    rewards2, dlogps2, samples2, acceptance_rates2 = my_chains.run_episode()

    for x in np.linspace(-6.0, 6.0, 20):
        m_old, sd_old = my_policy.get_proposal_distribution(np.array([x]), None, old_params)
        m_new, sd_new = my_policy.get_proposal_distribution(np.array([x]), None, my_policy.params)
        print "position: {0} mean, sd: {1}, {2}".format(x, m_new - m_old, sd_new - sd_old)

    grads = np.zeros((25, 2))
    grads_bbvi = np.zeros((25, 2))
    for i in range(25):
        s = i*10
        e = (i+1)*10
        rs = rewards[s:e]
        ds = []
        for p in dlogps:
            ds.append(p[s:e])
        g = vimco.estimate_gradient(rs, ds)
        grads[i, 0] = g[0][0][0]
        grads[i, 1] = g[1][0]

        g = bbvi.estimate_gradient(rs, ds)
        grads_bbvi[i, 0] = g[0][0][0]
        grads_bbvi[i, 1] = g[1][0]

