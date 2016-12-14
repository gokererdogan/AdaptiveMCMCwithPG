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

from rllib.environment import Environment, MHEnvironment
from rllib.space import FiniteActionSpace, MHStateSpace
from rllib.parameter_schedule import GreedyEpsilonLinearSchedule, GreedyEpsilonConstantSchedule
from rllib.q_learning import QNeuralNetwork, QLearningAgent

from occluding_tiles_problem import *


class OccludingTilesActionSpace(FiniteActionSpace):
    def __init__(self):
        actions = ['move', 'resize', 'rotate']
        FiniteActionSpace.__init__(self, actions)

    def reverse(self, action):
        return action


class OccludingTilesEnvironment(MHEnvironment):
    def __init__(self, state_space, action_params):
        Environment.__init__(self, state_space)

        self.action_params = action_params

    def _apply_action_to_hypothesis(self, hypothesis, action):
        """
        if action == "add/remove":
            new_state, _, _ = add_remove_tile(self.current_state, params=self.action_params)
        """
        if action == "move":
            new_state, _, _ = move_tile(hypothesis, params=self.action_params)
        elif action == "resize":
            new_state, _, _ = resize_tile(hypothesis, params=self.action_params)
        elif action == "rotate":
            new_state, _, _ = rotate_tile(hypothesis, params=self.action_params)
        else:
            raise ValueError("Unknown action.")

        return new_state


if __name__ == "__main__":
    epoch_count = 10
    decrease_eps_for = 5
    episodes_per_epoch = 10
    episode_length = 100
    likelihood_variance = 0.1
    learning_rate = 0.001

    eps_schedule = GreedyEpsilonConstantSchedule(eps=0.2)

    max_tile_count = 1
    proposal_params = {'MAX_TILE_COUNT': max_tile_count}

    ot_action_space = OccludingTilesActionSpace()
    ot_state_space = MHStateSpace(hypothesis_class=OccludingTilesHypothesis,
                                  data=OccludingTilesHypothesis(tile_count=1).render(),
                                  reward_type='acceptance',
                                  ll_variance=likelihood_variance)
    env = OccludingTilesEnvironment(ot_state_space, proposal_params)

    # build neural network
    action_dim = len(ot_action_space)
    nn = lasagne.layers.InputLayer(shape=(1,) + IMG_SIZE)
    nn = lasagne.layers.DenseLayer(incoming=nn, num_units=500, nonlinearity=lasagne.nonlinearities.rectify)
    nn = lasagne.layers.DenseLayer(incoming=nn, num_units=action_dim, W=lasagne.init.Normal(0.01), b=None,
                                   nonlinearity=lasagne.nonlinearities.linear)

    q_function = QNeuralNetwork(nn, ot_state_space, ot_action_space, learning_rate=learning_rate)
    q_learner = QLearningAgent(q_function, discount_factor=0.98, greed_eps=eps_schedule)

    epoch_rewards = np.zeros(epoch_count)
    epoch_log_p = np.zeros(epoch_count)
    epoch_acceptance_rate = np.zeros(epoch_count)

    for e in range(epoch_count):
        for i in range(episodes_per_epoch):
            progress_bar(i + 1, max=episodes_per_epoch, update_freq=episodes_per_epoch / 100 or 1)
            env.set_observed_data(OccludingTilesHypothesis(tile_count=1, ll_variance=likelihood_variance).render())
            states, actions, rewards = env.run(agent=q_learner, episode_length=episode_length)
            if np.any(np.isnan(q_function.nn.W.get_value())):
                raise RuntimeError("Nan encountered in W.")
            epoch_rewards[e] += np.sum(rewards)
            epoch_log_p[e] += np.sum([state[0].log_likelihood(env.state_space.data) + state[0].log_prior()
                                      for state in states])
            epoch_acceptance_rate[e] += np.mean([state[1] for state in states])
        epoch_rewards[e] /= episodes_per_epoch
        epoch_log_p[e] /= episodes_per_epoch
        epoch_acceptance_rate[e] /= episodes_per_epoch
        print("Epoch {0:d}| Avg. reward per episode: {1:f}; "
              "Avg. log prob.: {2:f}; "
              "Avg. acceptance rate: {3:.4f}%; ".format(e + 1, epoch_rewards[e], epoch_log_p[e],
                                                        epoch_acceptance_rate[e] * 100))

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
        print q_function.get_q((s3, None, None))

        s4 = h.copy()
        s4.tiles[0].size[0] *= 0.8
        s4.tiles[0].size[1] *= 1.2
        print "resize"
        print q_function.get_q((s4, None, None))

        s5 = h.copy()
        s5.tiles[0].orientation += np.pi / 2.0
        print "rotate"
        print q_function.get_q((s5, None, None))
