import argparse
import numpy as np

import occluding_tiles_problem as otp

import lddp_sampler
import reward


def run_chain(run_id, seed, episode_length, episode_count, sample_freq, policy_sd='low',
              save_folder="./results/evaluations"):
    assert episode_count % sample_freq == 0, "Episode count must be a multiple of sample frequency."

    # write run parameters to file
    with open("{0:s}/run_log.txt".format(save_folder), "a") as f:
        f.write("{0:s} {1:d} {2:d} {3:d} {4:d} {5:s}\n".format(run_id, seed, episode_length, episode_count,
                                                               sample_freq, policy_sd))

    np.random.seed(seed)

    chain_count = 20
    my_target = otp.OccludingTilesDistribution(ll_sd=0.02)

    # save ground truth
    np.save("{0:s}/{1:d}_ground_truth.npy".format(save_folder, seed), my_target.true_x)

    if policy_sd == 'low':
        move_sd = 1.0
    else:
        move_sd = 2.0
    my_policy = otp.OccludingTilesPolicy(learn_pick_tile=False, learn_move_tile=True,
                                         move_filter_count=50, move_filter_size=(3, 3),
                                         move_pool_size=(4, 4), move_sd_multiplier=move_sd)

    file_prefix = '{0:s}_{1:d}_{2:s}'.format(run_id, seed, policy_sd)
    if run_id == "naive":
        for i in range(len(my_policy.params)):
            my_policy.params[i][:] = 0.0
    else:
        learned_params = np.load('results/runs/{0:s}_params.npy'.format(run_id))
        for i in range(len(my_policy.params)):
            my_policy.params[i] = learned_params[-1][i]

    my_chains = lddp_sampler.ParallelMHChains(target_distribution=my_target, policy=my_policy,
                                              reward_function=reward.log_prob_increase_avg, chain_count=chain_count,
                                              episode_length=episode_length, thinning_period=episode_length)

    sample_count = episode_count / sample_freq
    samples = np.zeros((sample_count, chain_count) + my_chains.x0[0].shape)
    rewards = np.zeros((episode_count, chain_count))
    log_lls = np.zeros((episode_count, chain_count))
    rmse = np.zeros((episode_count, chain_count))
    acceptance_rates = np.zeros((episode_count, chain_count))
    for e in range(episode_count):
        e_rewards, _, e_samples, e_acceptance_rates = my_chains.run_episode()
        for c in range(chain_count):
            rewards[e, c] = e_rewards[c]
            log_lls[e, c] = my_target.log_probability(e_samples[c][-1])
            rmse[e, c] = np.sqrt(np.mean(np.square((my_target.true_x - e_samples[c][-1]) / otp.X_BOUND)))
            acceptance_rates[e, c] = e_acceptance_rates[c]
        if (e + 1) % sample_freq == 0:
            for c in range(chain_count):
                samples[(e+1)/sample_freq - 1, c] = e_samples[c][-1]
        print e, log_lls[e]

    np.save('{0:s}/{1:s}_lls.npy'.format(save_folder, file_prefix), log_lls)
    np.save('{0:s}/{1:s}_rmse.npy'.format(save_folder, file_prefix), rmse)
    np.save('{0:s}/{1:s}_ars.npy'.format(save_folder, file_prefix), acceptance_rates)
    np.save('{0:s}/{1:s}_samples.npy'.format(save_folder, file_prefix), samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", help="Run id (6 digit number) of the proposal to run. "
                                       "Use 'naive' for naive proposal.")
    parser.add_argument("episode_length", type=int, help="Episode length")
    parser.add_argument("episode_count", type=int, help="Episode count")
    parser.add_argument("sample_freq", type=int, help="Sample frequency")
    parser.add_argument("move_sd", type=str, choices=['low', 'high'], default='low',
                        help="Standard deviation of proposal.")
    parser.add_argument("seed", type=int, nargs='+', help="Random number seeds.")
    args = parser.parse_args()
    for s in args.seed:
        print args.run_id, s
        run_chain(run_id=args.run_id, seed=s, episode_length=args.episode_length, episode_count=args.episode_count,
                  sample_freq=args.sample_freq, policy_sd=args.move_sd)

