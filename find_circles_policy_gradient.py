"""
Learning data-driven proposals through reinforcement learning

Policy gradient for find circles problem. The purpose here is to directly learn a
policy (probability of picking each move given a state) to use as a data-driven
proposal strategy.

5 Aug 2016
https://github.com/gokererdogan
"""
import lasagne
from lasagne.updates import sgd

from gmllib.helpers import progress_bar

from rllib.parameter_schedule import GreedyEpsilonConstantSchedule
from rllib.policy_gradient import PolicyGradientAgent, PolicyNeuralNetworkMultinomial
from rllib.space import FiniteActionSpace

from find_circles_problem import *
from find_circles_q_learning import FindCirclesEnvironment


if __name__ == "__main__":
    epoch_count = 5
    episodes_per_epoch = 1000
    likelihood_variance = 0.1

    eps_schedule = GreedyEpsilonConstantSchedule(eps=0.1)

    env = FindCirclesEnvironment(data=FindCirclesProblem().render(), ll_variance=likelihood_variance)
    action_space = FiniteActionSpace(actions=[0, 1, 2, 3, 4])

    # build neural network
    nn = lasagne.layers.InputLayer(shape=(1,) + IMG_SIZE)
    nn = lasagne.layers.DenseLayer(incoming=nn, num_units=5, W=lasagne.init.Normal(0.01), b=None,
                                   nonlinearity=lasagne.nonlinearities.softmax)

    policy_function = PolicyNeuralNetworkMultinomial(nn, env.state_space, action_space, learning_rate=0.001,
                                                     optimizer=sgd)
    pg_learner = PolicyGradientAgent(policy_function, discount_factor=0.9, greed_eps=eps_schedule, update_freq=100)

    rewards = np.zeros(epoch_count)

    for e in range(epoch_count):
        for i in range(episodes_per_epoch):
            progress_bar(i+1, max=episodes_per_epoch, update_freq=episodes_per_epoch/100)
            env.set_observed_data(FindCirclesProblem(ll_variance=likelihood_variance).render())
            s, a, r = env.run(pg_learner, 10)
            rewards[e] += np.sum(r)
        rewards[e] /= episodes_per_epoch
        print("Epoch {0:d}| Avg. reward per episode: {1:f}".format(e+1, rewards[e]))

    s = FindCirclesProblem(configuration=[True, True, True, True, True])
    env.set_observed_data(s.render())
    s.configuration[0] = False
    print policy_function._forward(env.state_space.to_vector(s)[np.newaxis, :])
    s.configuration[1] = False
    print policy_function._forward(env.state_space.to_vector(s)[np.newaxis, :])


