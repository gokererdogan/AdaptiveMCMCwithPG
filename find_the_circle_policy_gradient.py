"""
Learning data-driven proposals through reinforcement learning

Policy gradient for find the circle problem. The purpose here is to learn a
policy function (producing a 2D vector specifying how much we should move the
circle given a state) to use as a data-driven proposal strategy.

21 Sept. 2016
https://github.com/gokererdogan
"""
import theano
import lasagne
from lasagne.updates import sgd, adagrad

from gmllib.helpers import progress_bar

from rllib.parameter_schedule import GreedyEpsilonConstantSchedule
from rllib.policy_gradient import PolicyGradientAgent, PolicyNeuralNetworkNormal
from rllib.environment import Environment
from rllib.space import StateSpace, RealActionSpace

from find_the_circle_problem import *


class FindTheCircleStateSpace(StateSpace):
    def __init__(self, data, ll_variance):
        StateSpace.__init__(self)
        self.data = data
        self.ll_variance = ll_variance

    def get_initial_state(self):
        state = FindTheCircleProblem(ll_variance=self.ll_variance)
        return state

    def get_reward(self, state):
        return state.log_likelihood(self.data)

    def is_goal_state(self, state):
        # if the state vector is a blank image, i.e., predicted and observed are the same
        if np.sum(np.square(self.to_vector(state))) < 20.0:  # less than 20 pixels are different
            return True
        return False

    def to_string(self, state):
        return str(state)

    def to_vector(self, state):
        # vector representation of the state is the difference image, i.e., (prediction - observed)
        image = state.render()
        x = (image - self.data).astype(theano.config.floatX)
        # add one dimension for number of channels
        return x[np.newaxis, :]


class FindTheCircleActionSpace(RealActionSpace):
    def __init__(self):
        RealActionSpace.__init__(self)

    def shape(self):
        return 2,

    def to_string(self, action):
        return str(action)

    def to_vector(self, action):
        return action


class FindTheCircleEnvironment(Environment):
    def __init__(self, data, ll_variance):
        state_space = FindTheCircleStateSpace(data, ll_variance)
        Environment.__init__(self, state_space)

    def _advance(self, action):
        new_state, _, _ = move_circle(self.current_state, params=None, move=action)
        return new_state

    def set_observed_data(self, data):
        self.state_space.data = data

if __name__ == "__main__":
    epoch_count = 25
    episodes_per_epoch = 1000
    episode_length = 50
    likelihood_variance = 0.1

    eps_schedule = GreedyEpsilonConstantSchedule(eps=0.0)

    env = FindTheCircleEnvironment(data=FindTheCircleProblem().render(), ll_variance=likelihood_variance)
    action_space = FindTheCircleActionSpace()

    # build neural network
    action_dim = int(np.prod(action_space.shape()))
    nn = lasagne.layers.InputLayer(shape=(1, 1) + IMG_SIZE)
    nn = lasagne.layers.DenseLayer(incoming=nn, num_units=action_dim, W=lasagne.init.Normal(0.01), b=None,
                                   nonlinearity=lasagne.nonlinearities.linear)

    policy_function = PolicyNeuralNetworkNormal(nn, env.state_space, action_space, learning_rate=0.1,
                                                optimizer=adagrad, cov_type='identity', std_dev=5.0)
    pg_learner = PolicyGradientAgent(policy_function, discount_factor=0.98, greed_eps=eps_schedule, update_freq=100)

    rewards = np.zeros(epoch_count)
    goals_reached = np.zeros(epoch_count)

    for e in range(epoch_count):
        for i in range(episodes_per_epoch):
            progress_bar(i+1, max=episodes_per_epoch, update_freq=episodes_per_epoch/100)
            env.set_observed_data(FindTheCircleProblem(ll_variance=likelihood_variance).render())
            s, a, r = env.run(pg_learner, episode_length)
            if np.any(np.isnan(policy_function.nn.W.get_value())):
                raise ValueError("nan encountered in W.")
            if len(s) <= episode_length:
                goals_reached[e] += 1.0
            rewards[e] += np.sum(r)
        rewards[e] /= episodes_per_epoch
        goals_reached[e] /= episodes_per_epoch
        print("Epoch {0:d}| Avg. reward per episode: {1:f}; Goal reached {2:f}%".format(e+1, rewards[e],
                                                                                        goals_reached[e]))

    h = FindTheCircleProblem(ll_variance=likelihood_variance)
    env.set_observed_data(h.render())
    print "Observed:"
    print h

    s1 = h.copy()
    s1.position[0] += 5.0
    s1.position[1] -= 10.0
    print "Current hypothesis"
    print s1

    print "Proposed move mean"
    print policy_function._forward(env.state_space.to_vector(s1)[np.newaxis, :])

    s2 = h.copy()
    s2.position[0] -= 5.0
    s2.position[1] += 10.0
    print "Current hypothesis"
    print s2

    print "Proposed move mean"
    print policy_function._forward(env.state_space.to_vector(s2)[np.newaxis, :])

    s3 = h.copy()
    s3.position[0] += 15.0
    s3.position[1] += 5.0
    print "Current hypothesis"
    print s3

    print "Proposed move mean"
    print policy_function._forward(env.state_space.to_vector(s3)[np.newaxis, :])
