"""
Learning data-driven proposals through reinforcement learning

Q-learning for occluding tiles problem. The purpose here is to learn a value function
to use as a data-driven proposal strategy.

24 Aug 2016
https://github.com/gokererdogan
"""
import numpy as np
import theano
import lasagne

from gmllib.helpers import progress_bar

from rllib.environment import Environment
from rllib.space import StateSpace, FiniteActionSpace
from rllib.parameter_schedule import GreedyEpsilonLinearSchedule, GreedyEpsilonConstantSchedule
from rllib.q_learning import QNeuralNetwork, QLearningAgent

from occluding_tiles_problem import *


class OccludingTilesStateSpace(StateSpace):
    def __init__(self, data, ll_variance):
        StateSpace.__init__(self)
        self.data = data
        self.ll_variance = ll_variance

    def get_initial_state(self):
        state = OccludingTilesHypothesis(ll_variance=self.ll_variance)
        return state

    def get_reward(self, state):
        return state.log_likelihood(self.data)

    def is_goal_state(self, state):
        # if the state vector is a blank image, i.e., predicted and observed are the same
        if np.sum(np.abs(self.to_vector(state))) < 10.0:  # less than 10 pixels are different
            return True
        return False

    def to_string(self, state):
        return str(state)

    def to_vector(self, state):
        # vector representation of the state is the difference image, i.e., (prediction - observed)
        image = state.render()
        x = (image - self.data).astype(theano.config.floatX)
        return x[np.newaxis, :]


class OccludingTilesEnvironment(Environment):
    def __init__(self, data, ll_variance, action_params):
        state_space = OccludingTilesStateSpace(data, ll_variance)
        Environment.__init__(self, state_space)

        self.action_params = action_params

    def _advance(self, action):
        """
        if action == "add/remove":
            new_state, _, _ = add_remove_tile(self.current_state, params=self.action_params)
        """
        if action == "move":
            new_state, _, _ = move_tile(self.current_state, params=self.action_params)
        elif action == "resize":
            new_state, _, _ = resize_tile(self.current_state, params=self.action_params)
        elif action == "rotate":
            new_state, _, _ = rotate_tile(self.current_state, params=self.action_params)
        else:
            raise ValueError("Unknown action.")

        """
        if self.state_space.get_reward(new_state) > self.state_space.get_reward(self.current_state):
            return new_state
        else:
            return self.current_state
        """
        return new_state

    def set_observed_data(self, data):
        self.state_space.data = data


if __name__ == "__main__":
    epoch_count = 10
    decrease_eps_for = 5
    episodes_per_epoch = 10
    episode_length = 100
    likelihood_variance = 0.1
    learning_rate = 0.001

    # eps_schedule = GreedyEpsilonConstantSchedule(eps=0.2)
    eps_schedule = GreedyEpsilonLinearSchedule(start_eps=1.0, end_eps=0.2,
                                               no_episodes=decrease_eps_for*episodes_per_epoch,
                                               decrease_period=episodes_per_epoch)

    max_tile_count = 1
    action_params = {'MAX_TILE_COUNT': max_tile_count}
    env = OccludingTilesEnvironment(data=OccludingTilesHypothesis(tile_count=1).render(), ll_variance=likelihood_variance,
                                    action_params=action_params)
    action_space = FiniteActionSpace(actions=['move', 'resize', 'rotate'])

    # build neural network
    action_dim = len(action_space)
    nn = lasagne.layers.InputLayer(shape=(1, 1) + IMG_SIZE)
    """
    nn = lasagne.layers.Conv2DLayer(incoming=nn, filter_size=(10, 10), num_filters=16, stride=(5, 5),
                                    nonlinearity=lasagne.nonlinearities.rectify)
    """
    nn = lasagne.layers.DenseLayer(incoming=nn, num_units=500, nonlinearity=lasagne.nonlinearities.rectify)
    nn = lasagne.layers.DenseLayer(incoming=nn, num_units=action_dim, W=lasagne.init.Normal(0.01), b=None,
                                   nonlinearity=lasagne.nonlinearities.linear)

    q_function = QNeuralNetwork(nn, env.state_space, action_space, learning_rate=learning_rate)
    q_learner = QLearningAgent(q_function, discount_factor=0.98, greed_eps=eps_schedule)

    goals_reached = np.zeros(epoch_count)
    rewards = np.zeros(epoch_count)

    for e in range(epoch_count):
        for i in range(episodes_per_epoch):
            progress_bar(i+1, max=episodes_per_epoch, update_freq=episodes_per_epoch/100 or 1)
            env.set_observed_data(OccludingTilesHypothesis(tile_count=1, ll_variance=likelihood_variance).render())
            s, a, r = env.run(q_learner, episode_length)
            if np.any(np.isnan(q_function.nn.W.get_value())):
                raise ValueError("nan encountered in W.")
            if len(s) <= episode_length:
                goals_reached[e] += 1.0
            rewards[e] += np.sum(r)
        rewards[e] /= episodes_per_epoch
        goals_reached[e] /= episodes_per_epoch
        print("Epoch {0:d}| Avg. reward per episode: {1:f}; Goal reached {2:4f}%".format(e+1, rewards[e],
                                                                                         goals_reached[e]))

    # test q_function
    h = OccludingTilesHypothesis(tile_count=1)
    env.set_observed_data(h.render())

    """
    s1 = h.copy()
    del s1.tiles[0]
    print "add"
    print q_function.get_q(s1)

    s2 = h.copy()
    s2.tiles.append(Tile())
    print "remove"
    print q_function.get_q(s2)
    """

    s3 = h.copy()
    s3.tiles[0].position[0] += 5.0
    s3.tiles[0].position[1] += -5.0
    print "move"
    print q_function.get_q(s3)

    s4 = h.copy()
    s4.tiles[0].size[0] *= 0.8
    s4.tiles[0].size[1] *= 1.2
    print "resize"
    print q_function.get_q(s4)

    s5 = h.copy()
    s5.tiles[0].orientation += np.pi / 2.0
    print "rotate"
    print q_function.get_q(s5)
