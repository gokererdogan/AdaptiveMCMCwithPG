"""
Learning data-driven proposals through reinforcement learning

Find the circle problem with data-driven proposal.

21 September 2016
https://github.com/gokererdogan
"""
import lasagne

from rllib.space import MHStateSpace
from rllib.policy_gradient import PolicyNeuralNetworkNormal, PolicyGradientAgent

from find_the_circle_problem import *
from find_the_circle_policy_gradient import FindTheCircleActionSpace


def data_driven_move_circle_move(h, params, agent):
    # get action
    state = (h, None, None)
    action = agent.get_action(state, agent.action_space)

    hp = move_circle(h, action)

    p_sp_s = agent.get_action_probability(state, action)
    reverse_action = agent.action_space.reverse(action)
    new_state = (hp, None, None)  # remember what state is comprised of!
    p_s_sp = agent.get_action_probability(new_state, reverse_action)

    return hp, p_sp_s, p_s_sp


if __name__ == "__main__":
    likelihood_variance = 0.001
    move_std_dev = 5.0

    ftc_action_space = FindTheCircleActionSpace()
    ftc_state_space = MHStateSpace(hypothesis_class=FindTheCircleProblem, data=FindTheCircleProblem().render(),
                                   reward_type='acceptance', ll_variance=likelihood_variance)

    # build neural network
    action_dim = int(np.prod(ftc_action_space.shape()))
    nn = lasagne.layers.InputLayer(shape=(1,) + IMG_SIZE)
    nn = lasagne.layers.DenseLayer(incoming=nn, num_units=action_dim, W=lasagne.init.Normal(0.01), b=None,
                                   nonlinearity=lasagne.nonlinearities.linear)
    w = np.load('results/ftc_W.npy')
    nn.W.set_value(w)

    policy_function = PolicyNeuralNetworkNormal(nn, ftc_state_space, ftc_action_space, learning_rate=0.0001,
                                                optimizer=lasagne.updates.sgd,
                                                cov_type='identity', std_dev=move_std_dev)
    pg_learner = PolicyGradientAgent(policy_function, discount_factor=0.98, update_freq=10)

    import mcmclib.proposal

    moves = {'data_driven_move_circle': lambda h, p: data_driven_move_circle_move(h, p, pg_learner)}
    proposal = mcmclib.proposal.RandomMixtureProposal(moves, params=None)

    import mcmclib.mh_sampler

    run_count = 20
    sample_count = 5
    thinning_period = 50
    sampler_class = 'mh'
    runs = []
    avg_log_prob = np.zeros(sample_count*thinning_period)
    for r in range(run_count):
        h = FindTheCircleProblem(ll_variance=likelihood_variance)
        observed = FindTheCircleProblem(ll_variance=likelihood_variance).render()
        sampler = mcmclib.mh_sampler.MHSampler(h, observed, proposal, burn_in=0, sample_count=sample_count,
                                               best_sample_count=sample_count, thinning_period=thinning_period,
                                               report_period=thinning_period)
        run = sampler.sample()
        runs.append(run)
        avg_log_prob += run.run_log.LogProbability

    avg_log_prob /= run_count
