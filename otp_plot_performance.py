from init_plotting import *
import numpy as np

import occluding_tiles_problem as otp

target_distribution = otp.OccludingTilesDistribution(ll_sd=0.02)


class RunResult(object):
    def __init__(self, run_id, move_sd, seeds, results_folder="./results/evaluations"):
        self.run_id = run_id
        self.seeds = seeds
        self.move_sd = move_sd
        ll_files = ["{0:s}/{1:s}_{2:d}_{3:s}_lls.npy".format(results_folder, run_id, seed, move_sd) for seed in seeds]
        ar_files = ["{0:s}/{1:s}_{2:d}_{3:s}_ars.npy".format(results_folder, run_id, seed, move_sd) for seed in seeds]
        rmse_files = ["{0:s}/{1:s}_{2:d}_{3:s}_rmse.npy".format(results_folder, run_id, seed, move_sd) for seed in seeds]
        sample_files = ["{0:s}/{1:s}_{2:d}_{3:s}_samples.npy".format(results_folder, run_id, seed, move_sd) for seed in seeds]
        self.lls = np.zeros((len(seeds),) + np.load(ll_files[0]).shape)
        self.ars = np.zeros((len(seeds),) + np.load(ar_files[0]).shape)
        self.rmses = np.zeros((len(seeds),) + np.load(rmse_files[0]).shape)
        self.samples = np.zeros((len(seeds),) + np.load(sample_files[0]).shape)
        for i in range(len(seeds)):
            self.lls[i] = np.load(ll_files[i])
            self.ars[i] = np.load(ar_files[i])
            self.rmses[i] = np.load(rmse_files[i])
            self.samples[i] = np.load(sample_files[i])


def plot_sample(run1, run2, seed, sample_ix, chain_ix="best"):
    gt_img = target_distribution.render_rgb(ground_truth[seed])
    run1_results = results[run1]
    run2_results = results[run2]
    run1_seed_ix = run1_results.seeds.index(seed)
    run2_seed_ix = run2_results.seeds.index(seed)
    episode_ix = (sample_ix + 1) * sample_freq - 1
    r1_lls = run1_results.lls[run1_seed_ix, episode_ix]
    r2_lls = run2_results.lls[run2_seed_ix, episode_ix]
    if chain_ix == "best":
        run1_chain_ix = np.argmax(r1_lls)
        run2_chain_ix = np.argmax(r2_lls)
    elif chain_ix == "mean":
        run1_chain_ix = np.argmin(np.abs(r1_lls - np.mean(r1_lls)))
        run2_chain_ix = np.argmin(np.abs(r2_lls - np.mean(r2_lls)))
    elif chain_ix == "median":
        run1_chain_ix = np.argmin(np.abs(r1_lls - np.median(r1_lls)))
        run2_chain_ix = np.argmin(np.abs(r2_lls - np.median(r2_lls)))
    else:
        run1_chain_ix = chain_ix
        run2_chain_ix = chain_ix
    run1_img = target_distribution.render_rgb(run1_results.samples[run1_seed_ix, sample_ix, run1_chain_ix])
    run2_img = target_distribution.render_rgb(run2_results.samples[run2_seed_ix, sample_ix, run2_chain_ix])
    f = plt.figure()
    ax_gt = f.add_axes([0.05, 0.25, 0.4, 0.4])
    ax_gt.imshow(gt_img)
    ax_gt.set_title("Ground truth")
    ax_gt.set_axis_off()
    ax1 = f.add_axes([0.55, 0.5, 0.4, 0.4])
    ax1.imshow(run1_img)
    ax1.set_title("{0:s}, {1:.2f}".format(run1, r1_lls[run1_chain_ix]))
    ax1.set_axis_off()
    ax2 = f.add_axes([0.55, 0.0, 0.4, 0.4])
    ax2.imshow(run2_img)
    ax2.set_title("{0:s}, {1:.2f}".format(run2, r2_lls[run2_chain_ix]))
    ax2.set_axis_off()
    plt.savefig("{0:s}/samples/sample_{1:d}_{2:s}_{3:d}.png".format(results_folder, seed, str(chain_ix),
                                                                    (episode_ix + 1) * episode_length))
    plt.close(f)


def plot_lls(agg_fun):
    plt.figure()
    for run_id, result in results.iteritems():
        lls = result.lls
        plt.plot(episode_length * np.arange(lls.shape[1]), agg_fun(lls, axis=(0, 2)), label="{0:s}".format(run_id))
    plt.legend(loc='best')
    plt.xlabel("Iteration")
    plt.ylabel("Log Likelihood")
    plt.savefig("{0:s}/lls_{1:s}_{2:s}.png".format(results_folder, result.move_sd, agg_fun.func_name))
    plt.close()


def plot_rmses(agg_fun):
    plt.figure()
    for run_id, result in results.iteritems():
        rmses = result.rmses
        plt.plot(episode_length * np.arange(rmses.shape[1]), agg_fun(rmses, axis=(0, 2)), label="{0:s}".format(run_id))
    plt.legend(loc='best')
    plt.xlabel("Iteration")
    plt.ylabel("RMSE")
    plt.savefig("{0:s}/rmses_{1:s}_{2:s}.png".format(results_folder, result.move_sd, agg_fun.func_name))
    plt.close()


if __name__ == "__main__":
    episode_length = 25
    sample_freq = 4
    seeds = [3, 78, 172, 5663, 45903]
    # runs = ["naive", "156475", "916956", "997608", "118178", "287129"]
    runs = ["naive", "156475"]
    move_sd = 'low'
    results_folder = "./results/evaluations"
    results = {run_id: RunResult(run_id, move_sd, seeds, results_folder) for run_id in runs}

    ground_truth = {seed: np.load("{0:s}/{1:d}_ground_truth.npy".format(results_folder, seed)) for seed in seeds}

    # print avg. acceptance rates
    print("Avg. acceptance rates")
    for run_id, result in results.iteritems():
        print("{0:s}: {1:f}".format(run_id, np.mean(result.ars)))

    plot_lls(np.mean)
    plot_lls(np.max)
    plot_lls(np.median)
    plot_rmses(np.mean)
    plot_rmses(np.max)
    plot_rmses(np.median)

    """
    for seed in [3]:
        print seed,
        for i in range(0, 50, 2):
            plot_sample("naive", "156475", seed=seed, sample_ix=i, chain_ix='mean')
            """
