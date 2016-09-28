"""
Learning data-driven proposals through reinforcement learning

Q-learning for find circles problem. The purpose here is to learn a value function
to use as a data-driven proposal strategy.

3 Aug 2016
https://github.com/gokererdogan
"""
import numpy as np
import theano

from gmllib.helpers import progress_bar

from rllib.environment import Environment
from rllib.space import StateSpace, FiniteActionSpace
from rllib.parameter_schedule import GreedyEpsilonLinearSchedule
from rllib.q_learning import QNeuralNetwork, QLearningAgent

from find_circles_problem import *


class FindCirclesStateSpace(StateSpace):
    def __init__(self, data, ll_variance):
        StateSpace.__init__(self)
        self.data = data
        self.ll_variance = ll_variance

    def get_initial_state(self):
        state = FindCirclesProblem(ll_variance=self.ll_variance)
        return state

    def get_reward(self, state):
        return state.log_likelihood(self.data)

    def is_goal_state(self, state):
        # if the state vector is a blank image, i.e., predicted and observed are the same
        if np.isclose(np.sum(self.to_vector(state) + 0.5), 0.0):
            return True
        return False

    def to_string(self, state):
        return str(state)

    def to_vector(self, state):
        # vector representation of the state is the difference image, i.e., (prediction - observed)
        image = state.render()
        x = (np.abs(image - self.data) - 0.5).astype(theano.config.floatX)
        return x


class FindCirclesEnvironment(Environment):
    def __init__(self, data, ll_variance):
        state_space = FindCirclesStateSpace(data, ll_variance)
        Environment.__init__(self, state_space)

    def _advance(self, action):
        new_state, _, _ = flip_circle(self.current_state, params=None, c_id=action)
        return new_state

    def set_observed_data(self, data):
        self.state_space.data = data


if __name__ == "__main__":
    epoch_count = 10
    decrease_eps_for = 5
    episodes_per_epoch = 500
    likelihood_variance = 0.1

    eps_schedule = GreedyEpsilonLinearSchedule(start_eps=1.0, end_eps=0.2,
                                               no_episodes=decrease_eps_for*episodes_per_epoch,
                                               decrease_period=episodes_per_epoch)

    env = FindCirclesEnvironment(data=FindCirclesProblem().render(), ll_variance=likelihood_variance)
    action_space = FiniteActionSpace(actions=[0, 1, 2, 3, 4])
    q_function = QNeuralNetwork([], env.state_space, action_space, learning_rate=0.001)
    q_learner = QLearningAgent(q_function, discount_factor=0.9, greed_eps=eps_schedule)

    rewards = np.zeros(epoch_count)

    for e in range(epoch_count):
        for i in range(episodes_per_epoch):
            progress_bar(i+1, max=episodes_per_epoch, update_freq=episodes_per_epoch/100)
            env.set_observed_data(FindCirclesProblem(ll_variance=likelihood_variance).render())
            s, a, r = env.run(q_learner, 250)
            rewards[e] += np.sum(r)
        rewards[e] /= episodes_per_epoch
        print("Epoch {0:d}| Avg. reward per episode: {1:f}".format(e+1, rewards[e]))

    # test q function
    s = FindCirclesProblem(configuration=[True, True, True, True, True])
    env.set_observed_data(s.render())
    s.configuration[0] = False
    print q_function.get_q(s)
    s.configuration[1] = False
    print q_function.get_q(s)


