import os
import argparse
import pickle
from model import *
from SALib.sample import saltelli
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore") 

# problem definition
PROBLEM = {
    "num_vars" : 5,
    "names"    : ["theta",
                  "gain_right",
                  "saving_prop",
                  "prob_left",
                  "alpha"],
    "bounds"   : [[0.05, 0.95],
                  [1.70, 8.00],
                  [0.70, 0.80],
                  [0.30, 0.45],
                  [2.00, 12.0]]
}

X = saltelli.sample(PROBLEM, 1024, calc_second_order=False)
UNIQUE_ALPHA = set(X[:,-1])
SEEDS = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
NUM_AGENTS = 1225
MU = 10
SIGMA = 1

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run SDA graph generation.")
    parser.add_argument("--graph_dir", type=str, help="The directory where SDA graph data is stored.")

    return parser.parse_args()

def main():
    args = parse_arguments()
    graph_dir = args.graph_dir    
    os.makedirs(graph_dir, exist_ok=True)

    for i in range(len(SEEDS)):
        # set random seed and generate initial wealth distribution
        s = SEEDS[i]
        random.seed(s)
        np.random.seed(s)
        w = np.random.normal(MU, SIGMA, NUM_AGENTS)
        
        # construct graph + extract communities for each unique alpha value
        for alpha in tqdm(UNIQUE_ALPHA):
            G = construct_sda_graph(w, alpha=alpha)
            communities = get_communities(G)
            community_membership = get_community_membership(G, communities)
            augmented_communities = get_augmented_communities(community_membership)
            with open(f"{graph_dir}/{i}_{alpha}.pickle", "wb") as f:
                pickle.dump((communities, community_membership, augmented_communities, w), f)

if __name__ == "__main__":
    main()
