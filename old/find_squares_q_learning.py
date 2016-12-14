"""
Learning data-driven proposals through reinforcement learning

Q-learning for find squares problem. The purpose here is to learn a value function
to use as a data-driven proposal strategy. Environment dynamics here are governed
by Metropolis-Hastings method; next state is accepted/rejected using MH rule.

3 Aug 2016
https://github.com/gokererdogan
"""
import theano
from lasagne.updates import adagrad

from gmllib.helpers import progress_bar

from rllib.environment import MHEnvironment
from rllib.space import FiniteActionSpace, MHStateSpace
from rllib.parameter_schedule import GreedyEpsilonConstantSchedule
from rllib.q_learning import QNeuralNetwork, QLearningAgent

from find_squares_problem import *


class FindSquaresStateSpace(MHStateSpace):
    def __init__(self, reward_type, **hypothesis_params):
        MHStateSpace.__init__(self, FindSquaresProblem, data=FindSquaresProblem().render(),
                              reward_type=reward_type, **hypothesis_params)

    def to_vector(self, state):
        # vector representation of the state is the absolute difference image, i.e., abs(prediction - observed)
        h = state['hypothesis']
        image = h.render()
        x = (np.abs(image - self.data) - 0.5).astype(theano.config.floatX)
        return x


class FindSquaresActionSpace(FiniteActionSpace):
    def __init__(self):
        actions = [(i, j) for i in range(SQUARES_PER_SIDE) for j in range(SQUARES_PER_SIDE)]
        FiniteActionSpace.__init__(self, actions)

    def reverse(self, action):
        # reverse of an action is itself
        return action


class FindSquaresMHEnvironment(MHEnvironment):
    def _apply_action_to_hypothesis(self, hypothesis, action):
        return flip_square(hypothesis, action)


if __name__ == "__main__":
    epoch_count = 20
    decrease_eps_for = 15
    episodes_per_epoch = 100
    episode_length = 250
    likelihood_variance = 0.01

    eps_schedule = GreedyEpsilonConstantSchedule(eps=0.2)

    fs_state_space = FindSquaresStateSpace(reward_type='log_p_increase', ll_variance=likelihood_variance)
    fs_action_space = FindSquaresActionSpace()

    q_function = QNeuralNetwork([], fs_state_space, fs_action_space, learning_rate=0.001, optimizer=adagrad)
    q_learner = QLearningAgent(q_function, discount_factor=0.98, greed_eps=eps_schedule)
    env = FindSquaresMHEnvironment(fs_state_space)

    epoch_rewards = np.zeros(epoch_count)
    epoch_log_p = np.zeros(epoch_count)
    epoch_acceptance_rate = np.zeros(epoch_count)

    for e in range(epoch_count):
        for i in range(episodes_per_epoch):
            progress_bar(i + 1, max=episodes_per_epoch, update_freq=episodes_per_epoch / 100 or 1)
            env.set_observed_data(FindSquaresProblem(ll_variance=likelihood_variance).render())
            states, actions, rewards = env.run(agent=q_learner, episode_length=episode_length)
            if np.any(np.isnan(q_function.nn.W.get_value())):
                raise RuntimeError("Nan encountered in W.")
            epoch_rewards[e] += np.sum(rewards)
            epoch_log_p[e] += np.sum([state['hypothesis'].log_likelihood(env.state_space.data) +
                                      state['hypothesis'].log_prior() for state in states])
            epoch_acceptance_rate[e] += np.mean([state['is_accepted'] for state in states])
        epoch_rewards[e] /= episodes_per_epoch
        epoch_log_p[e] /= episodes_per_epoch
        epoch_acceptance_rate[e] /= episodes_per_epoch
        print("Epoch {0:d}| Avg. reward per episode: {1:f}; "
              "Avg. log prob.: {2:f}; "
              "Avg. acceptance rate: {3:.4f}%; ".format(e + 1, epoch_rewards[e], epoch_log_p[e],
                                                        epoch_acceptance_rate[e] * 100))

        # test q function
        s = FindSquaresProblem(configuration=np.ones((SQUARES_PER_SIDE, SQUARES_PER_SIDE), dtype=bool))
        env.set_observed_data(s.render())
        s.configuration[0, 0] = False
        print q_function.get_q({'hypothesis': s})

        s.configuration[0, 1] = False
        print q_function.get_q({'hypothesis': s})
