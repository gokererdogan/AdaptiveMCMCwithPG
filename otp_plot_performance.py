import numpy as np

if __name__ == "__main__":
    episode_length = 25
    runs = ['naive']
    # run_ids = [528566, 641333, 686478, 812121, 955304, 58470]
    # run_ids = [641333, 812121, 58470]
    # run_ids = [365703, 780732]
    run_ids = [156475, 837720, 690281, 365703]
    for run_id in run_ids:
        runs.append("learned_{0:06d}".format(run_id))

    from init_plotting import *
    plt.figure()
    for run in runs:
        lls = np.load("results/evaluations/lls_{0:s}.npy".format(run))
        plt.plot(25 * np.arange(lls.shape[0]), np.mean(lls, axis=1), label=run)
    plt.legend(loc='best')
    plt.xlabel("Iteration")
    plt.ylabel("Log likelihood")
    plt.savefig("results/evaluations/lls.png")

    plt.figure()
    for run in runs:
        rmses = np.load("results/evaluations/rmse_{0:s}.npy".format(run))
        plt.plot(25 * np.arange(rmses.shape[0]), np.mean(rmses, axis=1), label=run)
    plt.legend(loc='best')
    plt.xlabel("Iteration")
    plt.ylabel("RMSE")
    plt.savefig("results/evaluations/rmse.png")
