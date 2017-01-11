import numpy as np

import find_the_dot_problem as ftd

import lddp_sampler
import reward
import gradient_estimator


if __name__ == "__main__":
    episode_length = 50
    data = ftd.FindTheDotTargetDistribution.render(np.zeros(1))
    my_target = ftd.FindTheDotTargetDistribution(ll_variance=0.2, data=data)

    naive_estimator = gradient_estimator.GradientEstimator(clip=False)
    vimco = gradient_estimator.VIMCOEstimator(clip=False)
    bbvi = gradient_estimator.BBVIEstimator(clip=False)

    my_policy = ftd.FindTheDotPolicy(type='linear')
    old_params = []
    for p in my_policy.params:
        p[:] = 0.01*np.random.randn(*p.shape)
        old_params.append(p.copy())

    chain_count = 1000
    reward_fn = reward.log_prob_increase_avg
    seed = np.random.randint(1000000)
    # seed = 834657
    print(1)
    my_chains = lddp_sampler.ParallelMHChains(target_distribution=my_target, policy=my_policy,
                                              reward_function=reward_fn,
                                              chain_count=chain_count,
                                              episode_length=episode_length, thinning_period=10)

    np.random.seed(seed)
    my_chains.reset()
    print("x0 < 0: {0}".format(np.sum(my_chains.x0 < 0.0)))
    print("x0 > 0: {0}".format(np.sum(my_chains.x0 > 0.0)))
    my_target.data = data
    rewards, dlogps, samples, acceptance_rates = my_chains.run_episode()
    grad_vimco = vimco.estimate_gradient(rewards, dlogps)
    grad_bbvi = bbvi.estimate_gradient(rewards, dlogps)
    grad_naive = naive_estimator.estimate_gradient(rewards, dlogps)

    mean_diffs = np.zeros((3, 10, len(ftd.XS)))
    lr = 1.0
    for i in range(10):
        s, e = i*(chain_count/10), (i+1)*(chain_count/10)
        ds = []
        for dlp in dlogps:
            ds.append(dlp[s:e])
        grad_vimco = vimco.estimate_gradient(rewards[s:e], ds)
        grad_bbvi = bbvi.estimate_gradient(rewards[s:e], ds)
        grad_naive = naive_estimator.estimate_gradient(rewards[s:e], ds)

        params_vimco = []
        for p, g in zip(my_policy.params, grad_vimco):
            params_vimco.append(p + lr*g)

        params_bbvi = []
        for p, g in zip(my_policy.params, grad_bbvi):
            params_bbvi.append(p + lr*g)

        params_naive = []
        for p, g in zip(my_policy.params, grad_naive):
            params_naive.append(p + lr*g)

        for xi, x in enumerate(ftd.XS):
            m_old, sd_old = my_policy.get_proposal_distribution(np.array([x]), my_target.data, old_params)
            m_new, sd_new = my_policy.get_proposal_distribution(np.array([x]), my_target.data, params_vimco)
            mean_diffs[0, i, xi] = m_new - m_old

            m_new, sd_new = my_policy.get_proposal_distribution(np.array([x]), my_target.data, params_bbvi)
            mean_diffs[1, i, xi] = m_new - m_old

            m_new, sd_new = my_policy.get_proposal_distribution(np.array([x]), my_target.data, params_naive)
            mean_diffs[2, i, xi] = m_new - m_old

    print "Done!"

