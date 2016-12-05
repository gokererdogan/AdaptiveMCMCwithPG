"""
Learning data-driven proposals through reinforcement learning

Policy gradient for find the circle problem. The purpose here is to learn a
policy function (producing a 2D vector specifying how much we should move the
circle given a state) to use as a data-driven proposal strategy.

21 Sept. 2016
https://github.com/gokererdogan
"""
import lasagne
from lasagne.updates import sgd, adagrad

from gmllib.helpers import progress_bar

from rllib.environment import MHEnvironment
from rllib.space import MHStateSpace
from rllib.policy_gradient import PolicyGradientAgent, PolicyNeuralNetworkNormal
from rllib.space import RealActionSpace

from find_the_circle_problem import *


class FindTheCircleActionSpace(RealActionSpace):
    def __init__(self):
        RealActionSpace.__init__(self)

    def shape(self):
        return 2,

    def reverse(self, action):
        # reverse of an action is simply -action
        return -action

    def to_string(self, action):
        return str(action)

    def to_vector(self, action):
        return action


class FindTheCircleEnvironment(MHEnvironment):
    def __init__(self, state_space):
        MHEnvironment.__init__(self, state_space)

    def _apply_action_to_hypothesis(self, hypothesis, action):
        return move_circle(hypothesis, action)


if __name__ == "__main__":
    epoch_count = 5
    episodes_per_epoch = 100
    episode_length = 50
    likelihood_variance = 0.0005
    move_std_dev = 10.0

    ftc_action_space = FindTheCircleActionSpace()
    ftc_state_space = MHStateSpace(hypothesis_class=FindTheCircleProblem, data=FindTheCircleProblem().render(),
                                   reward_type='acceptance', ll_variance=likelihood_variance)
    # build neural network
    action_dim = int(np.prod(ftc_action_space.shape()))
    nn = lasagne.layers.InputLayer(shape=(1,) + IMG_SIZE)
    nn = lasagne.layers.DenseLayer(incoming=nn, num_units=action_dim, W=lasagne.init.Normal(0.01), b=None,
                                   nonlinearity=lasagne.nonlinearities.linear)

    policy_function = PolicyNeuralNetworkNormal(nn, ftc_state_space, ftc_action_space, learning_rate=0.001,
                                                optimizer=sgd, cov_type='identity', std_dev=move_std_dev)
    pg_learner = PolicyGradientAgent(policy_function, discount_factor=0.98, update_freq=10)

    env = FindTheCircleEnvironment(ftc_state_space)

    epoch_rewards = np.zeros(epoch_count)
    epoch_log_p = np.zeros(epoch_count)
    epoch_acceptance_rate = np.zeros(epoch_count)

    for e in range(epoch_count):
        for i in range(episodes_per_epoch):
            progress_bar(i + 1, max=episodes_per_epoch, update_freq=episodes_per_epoch / 100 or 1)
            env.set_observed_data(FindTheCircleProblem(ll_variance=likelihood_variance).render())
            states, actions, rewards = env.run(agent=pg_learner, episode_length=episode_length)
            if np.any(np.isnan(policy_function.nn.W.get_value())):
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

        # test the policy function
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
        print policy_function._forward(env.state_space.to_vector((s1, None, None))[np.newaxis, :])

        s2 = h.copy()
        s2.position[0] -= 5.0
        s2.position[1] += 10.0
        print "Current hypothesis"
        print s2

        print "Proposed move mean"
        print policy_function._forward(env.state_space.to_vector((s2, None, None))[np.newaxis, :])

        s3 = h.copy()
        s3.position[0] += 15.0
        s3.position[1] += 5.0
        print "Current hypothesis"
        print s3

        print "Proposed move mean"
        print policy_function._forward(env.state_space.to_vector((s3, None, None))[np.newaxis, :])
