"""
Learning data-driven proposals through reinforcement learning

Policy gradient for find squares problem. The purpose here is to directly learn a
policy (probability of picking each move given a state) to use as a data-driven
proposal strategy.

5 Aug 2016
https://github.com/gokererdogan
"""
import lasagne
from lasagne.updates import adagrad

from gmllib.helpers import progress_bar

from rllib.policy_gradient import PolicyGradientAgent, PolicyNeuralNetworkMultinomial

from find_squares_problem import *
from find_squares_q_learning import FindSquaresMHEnvironment, FindSquaresActionSpace, FindSquaresStateSpace


if __name__ == "__main__":
    epoch_count = 5
    episodes_per_epoch = 200
    episode_length = 50
    likelihood_variance = 0.05

    fs_state_space = FindSquaresStateSpace(reward_type='acceptance', ll_variance=likelihood_variance)
    fs_action_space = FindSquaresActionSpace()

    nn = lasagne.layers.InputLayer(shape=(1,) + IMG_SIZE)
    nn = lasagne.layers.DenseLayer(incoming=nn, num_units=len(fs_action_space), W=lasagne.init.Normal(0.001),
                                   b=None, nonlinearity=lasagne.nonlinearities.softmax)
    policy_function = PolicyNeuralNetworkMultinomial(nn, fs_state_space, fs_action_space, learning_rate=0.001,
                                                     optimizer=adagrad)
    pg_learner = PolicyGradientAgent(policy_function, discount_factor=0.98, update_freq=10)
    env = FindSquaresMHEnvironment(fs_state_space)

    epoch_rewards = np.zeros(epoch_count)
    epoch_log_p = np.zeros(epoch_count)
    epoch_acceptance_rate = np.zeros(epoch_count)

    for e in range(epoch_count):
        for i in range(episodes_per_epoch):
            progress_bar(i + 1, max=episodes_per_epoch, update_freq=episodes_per_epoch / 100 or 1)
            env.set_observed_data(FindSquaresProblem(ll_variance=likelihood_variance).render())
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

        # test policy function
        s = FindSquaresProblem(configuration=np.ones((SQUARES_PER_SIDE, SQUARES_PER_SIDE), dtype=bool),
                               ll_variance=likelihood_variance)
        env.set_observed_data(s.render())
        s.configuration[0, 0] = False
        print policy_function.get_action_probability((s, None, None))

        s.configuration[0, 1] = False
        print policy_function.get_action_probability((s, None, None))


