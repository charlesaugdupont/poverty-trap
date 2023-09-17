import sys
import time
import lzma
import pickle
from model import *
from SALib.sample import saltelli


if __name__ == "__main__":

	SEEDS = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]

	# parse arguments
	output_dir = sys.argv[1]
	idx = int(sys.argv[2]) - 1
	seed_idx = int(sys.argv[3])
	SEED = SEEDS[seed_idx]

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

	# generate Saltelli samples
	NUM_SAMPLES = 1024
	X = saltelli.sample(PROBLEM, NUM_SAMPLES, calc_second_order=False)
	L = int(X.shape[0]/128)
	X = X[idx*L:(idx+1)*L]

	start_time = time.time()

	# run each param combination
	for iter_idx, row in enumerate(X):

		# load graph based on seed number and alpha parameter
		with open(f"../paper_sda_graphs/{seed_idx}_{row[4]}.pickle", "rb") as f:
			communities, community_membership, augmented_communities, initial_wealth = pickle.load(f)

		# compute project cost for each community based on theta parameter
		project_costs = get_community_project_costs(initial_wealth, augmented_communities, row[0])

		W, A, U, P, T, _, G = simulation (
			communities=communities,
			community_membership=community_membership,
			NUM_AGENTS=1250,
			STEPS=50,
			seed=SEED,
			PROJECT_COSTS=project_costs,
			gain_right=row[1],
			SAVING_PROP=row[2],
			prob_left=row[3],
			INIT_WEALTH_VALUES=initial_wealth
		)

		# store results
		data = {
			"W":W,
			"A":A,
			"U":U,
			"P":P,
			"T":np.array(list(T.values())).astype(np.int16),
			"G":G,
			"params":tuple(row),
		}
		pickle.dump(data, lzma.open(output_dir + f"/{seed_idx}_{idx*L + iter_idx + 1}_cpt_sda.pkl.lzma", 'wb'))

		print(f"JOB {idx} : finished seed {seed_idx}, param {idx*L + iter_idx + 1} at t = {(time.time() - start_time)/60:.0f} mins", 
			  flush=True)