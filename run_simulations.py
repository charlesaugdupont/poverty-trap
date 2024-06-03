from SALib.sample import saltelli
from model_alternate import *
import pickle
import time
import lzma
import sys


if __name__ == "__main__":

	SEEDS = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]

	# parse arguments
	output_dir = sys.argv[1]
	idx = int(sys.argv[2]) - 1
	seed_idx = int(sys.argv[3])
	SEED = SEEDS[seed_idx]

	# problem definition
	PROBLEM = {
		"num_vars" : 6,
		"names"    : ["theta",
					  "gain_right",
					  "saving_prop",
					  "prob_left",
					  "alpha",
					  "assistance"],
		"bounds"   : [[0.01, 0.20],
					  [1.70, 8.00],
					  [0.70, 0.80],
					  [0.30, 0.45],
					  [2.00, 12.0],
					  [0.05, 0.30]]
	}

	# generate Saltelli samples
	NUM_SAMPLES = 1024
	X = saltelli.sample(PROBLEM, NUM_SAMPLES, calc_second_order=False)
	L = int(X.shape[0]/128)
	X = X[idx*L:(idx+1)*L]

	start_time = time.time()

	# run each param combination
	for iter_idx, row in enumerate(X):

		# load graph based on seed number and alpha parameter
		with open(f"./sda_graphs/{seed_idx}_{row[4]}.pickle", "rb") as f:
			communities, community_membership, augmented_communities, initial_wealth = pickle.load(f)

		# compute project cost for each community based on theta parameter
		project_costs = get_community_project_costs(initial_wealth, augmented_communities, row[0])

		W, I, C, O, A, U, P, T, H = simulation (
			COMMUNITIES=communities,
			COMMUNITY_MEMBERSHIP=community_membership,
			SEED=SEED,
			PROJECT_COSTS=project_costs,
			GAIN_RIGHT=row[1],
			SAVING_PROP=row[2],
			PROB_LEFT=row[3],
			ASSISTANCE=row[5],
			INIT_WEALTH_VALUES=initial_wealth,
			idx=str(idx)
		)

		# store results
		data = {
			"W":W,
			"I":I,
			"C":C,
			"O":O,
			"A":A,
			"U":U,
			"P":P,
			"T":T,
			"H":H,
			"params":tuple(row)
		}
		pickle.dump(data, lzma.open(output_dir + f"/{seed_idx}_{idx*L + iter_idx + 1}_paper.pkl.lzma", 'wb'))

		print(f"JOB {idx} : finished seed {seed_idx}, param {idx*L + iter_idx + 1} at t = {(time.time() - start_time)/60:.0f} mins", flush=True)