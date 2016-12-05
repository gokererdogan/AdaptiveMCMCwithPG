"""
Learning data-driven proposals through reinforcement learning

Find squares problem with data-driven proposal function.

28 Sept. 2016
https://github.com/gokererdogan
"""
from rllib.q_learning import QNeuralNetwork, QLearningAgent
from rllib.parameter_schedule import GreedyEpsilonConstantSchedule
from rllib.space import MHStateSpace

from find_squares_q_learning import FindSquaresActionSpace, FindSquaresMHEnvironment
from find_squares_problem import *


def data_driven_flip_square_move(h, params, agent):
    # get action
    state = (h, None, None)
    action = agent.get_action(state, agent.action_space)
    hp = flip_square(h, action)

    p_sp_s = agent.get_action_probability(state, action)
    reverse_action = agent.action_space.reverse(action)
    new_state = (hp, None, None)  # remember what state is comprised of!
    p_s_sp = agent.get_action_probability(new_state, reverse_action)

    return hp, p_sp_s, p_s_sp


if __name__ == "__main__":
    likelihood_variance = 0.005

    fs_state_space = MHStateSpace(FindSquaresProblem, data=FindSquaresProblem().render(), reward_type='acceptance',
                                  ll_variance=likelihood_variance)
    fs_action_space = FindSquaresActionSpace()

    q_function = QNeuralNetwork([], fs_state_space, fs_action_space, learning_rate=0.001)
    W = np.load('results/find_squares_q_4x4_W_mh.npy')
    b = np.load('results/find_squares_q_4x4_b_mh.npy')
    q_function.nn.W.set_value(W)
    q_function.nn.b.set_value(b)

    eps_schedule = GreedyEpsilonConstantSchedule(eps=0.2)
    q_agent = QLearningAgent(q_function, discount_factor=0.98, greed_eps=eps_schedule)
    env = FindSquaresMHEnvironment(fs_state_space)

    import mcmclib.proposal

    moves = {'fc_flip_square_dd': lambda h, p: data_driven_flip_square_move(h, p, agent=q_agent)}
    proposal = mcmclib.proposal.RandomMixtureProposal(moves, params=None)

    import mcmclib.mh_sampler

    run_count = 20
    sample_count = 5
    thinning_period = 50
    sampler_class = 'mh'
    runs = []
    avg_log_prob = np.zeros(sample_count*thinning_period)
    for r in range(run_count):
        h = FindSquaresProblem(ll_variance=likelihood_variance)
        observed = FindSquaresProblem(ll_variance=likelihood_variance).render()
        fs_state_space.data = observed
        sampler = mcmclib.mh_sampler.MHSampler(h, observed, proposal, burn_in=0, sample_count=sample_count,
                                               best_sample_count=sample_count, thinning_period=thinning_period,
                                               report_period=thinning_period)
        run = sampler.sample()
        runs.append(run)
        avg_log_prob += run.run_log.LogProbability

    avg_log_prob /= run_count


