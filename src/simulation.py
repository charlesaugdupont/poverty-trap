import argparse
import pickle
import lzma
import os
from SALib.sample import saltelli
from model import *

import warnings
warnings.filterwarnings("ignore") 

# random seeds
SEEDS = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]

# problem definition
PROBLEM = {
    "num_vars": 5,
    "names" : ["theta",
               "gain_right",
               "saving_prop",
               "prob_left",
               "alpha"],
    "bounds": [[0.01, 0.20],
               [1.70, 8.00],
               [0.70, 0.80],
               [0.30, 0.45],
               [2.00, 12.0]]
}

NUM_SAMPLES = 1024

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the simulation with given parameters.")
    parser.add_argument("--graph_dir", type=str, help="The directory where SDA graph data is stored.")
    parser.add_argument("--output_dir", type=str, help="The directory where the output files will be saved.")
    parser.add_argument("--chunk_idx", type=int, help="Index for partitioning the set of parameter combinations (1-based).")
    parser.add_argument("--seed_idx", type=int, help="Index of the random seed to be used.")

    return parser.parse_args()

def main():
    args = parse_arguments()

    graph_dir = args.graph_dir
    output_dir = args.output_dir
    idx = args.chunk_idx - 1
    seed_idx = args.seed_idx
    seed = SEEDS[seed_idx]
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # generate Saltelli samples
    X = saltelli.sample(PROBLEM, NUM_SAMPLES, calc_second_order=False)
    print(f"\nTotal saltelli sample size is {X.shape}.")
    L = int(X.shape[0]/128)
    X = X[idx*L:(idx+1)*L]
    print(f"Script will run chunk #{idx+1} with {X.shape[0]} parameter combinations.\n")

    # run each param combination
    for iter_idx, row in enumerate(X):

        print(f"Running parameter combination {iter_idx+1}/{X.shape[0]}...")

        # load graph based on seed number and alpha parameter
        with open(f"{graph_dir}/{seed_idx}_{row[4]}.pickle", "rb") as f:
            communities, community_membership, augmented_communities, initial_wealth = pickle.load(f)

        # compute project cost for each community based on theta parameter
        project_costs = get_community_project_costs(initial_wealth, augmented_communities, row[0])

        W, I, C, O, A, U, P, T, H = simulation(
            COMMUNITIES=communities,
            COMMUNITY_MEMBERSHIP=community_membership,
            SEED=seed,
            PROJECT_COSTS=project_costs,
            GAIN_RIGHT=row[1],
            SAVING_PROP=row[2],
            PROB_LEFT=row[3],
            INIT_WEALTH_VALUES=initial_wealth
        )

        # store results
        data = {"W":W, "I":I, "C":C, "O":O, "A":A, "U":U, "P":P, "T":T, "H":H, "params":tuple(row)}
        pickle.dump(data, lzma.open(output_dir + f"/{seed_idx}_{idx*L + iter_idx + 1}_paper.pkl.lzma", 'wb'))

        print(f"JOB {idx} : finished seed {seed_idx}, param {idx*L + iter_idx + 1}.", flush=True)

if __name__ == "__main__":
    main()
