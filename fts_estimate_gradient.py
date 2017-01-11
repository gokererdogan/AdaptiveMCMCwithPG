import numpy as np

import find_the_square_problem as fts

import lddp_sampler
import reward
import gradient_estimator


if __name__ == "__main__":
    np.random.seed(1)

    episode_length = 100
    data = fts.FindTheSquareTargetDistribution.render(np.zeros(2))
    my_target = fts.FindTheSquareTargetDistribution(ll_variance=5.0, data=data)

    vimco = gradient_estimator.VIMCOEstimator(clip=False)
    bbvi = gradient_estimator.BBVIEstimator(clip=False)

    my_policy = fts.FindTheSquarePolicy(n_hidden=50)
    old_params = []
    for p in my_policy.params:
        p[:] = 0.01*np.random.randn(*p.shape)
        old_params.append(p.copy())

    chain_count = 250
    reward_fn = reward.log_prob_increase_avg
    seed = 49481
    print(1)
    my_chains = lddp_sampler.ParallelMHChains(target_distribution=my_target, policy=my_policy,
                                              reward_function=reward_fn,
                                              chain_count=chain_count,
                                              episode_length=episode_length, thinning_period=10)

    np.random.seed(seed)
    my_chains.x0 = 3.0 * np.ones((chain_count, 2))

    rewards, dlogps, samples, acceptance_rates = my_chains.run_episode()
    grad = vimco.estimate_gradient(rewards, dlogps)

    lr = 1.0
    for p, g in zip(my_policy.params, grad):
        p += lr * g

    # estimate reward with new params
    print(2)
    np.random.seed(seed)
    my_chains.x0 = 3.0 * np.ones((chain_count, 2))
    rewards2, dlogps2, samples2, acceptance_rates2 = my_chains.run_episode()

    x, y = 3.0, 3.0
    m_old, sd_old = my_policy.get_proposal_distribution(np.array([x, y]), my_target.data, old_params)
    m_new, sd_new = my_policy.get_proposal_distribution(np.array([x, y]), my_target.data, my_policy.params)
    print "position: {0}, {1} mean, sd: {2}, {3}".format(x, y, m_new - m_old, sd_new - sd_old)
    """
    for x in fts.XS[0::4]:
        for y in fts.YS[0::4]:
            m_old, sd_old = my_policy.get_proposal_distribution(np.array([x, y]), my_target.data, old_params)
            m_new, sd_new = my_policy.get_proposal_distribution(np.array([x, y]), my_target.data, my_policy.params)
            print "position: {0}, {1} mean, sd: {2}, {3}".format(x, y, m_new - m_old, sd_new - sd_old)
    """
