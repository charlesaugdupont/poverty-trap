import sys
import time
import lzma
import pickle
from model import *
from SALib.sample import saltelli


if __name__ == "__main__":

	SEEDS = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]

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
					  "beta",
					  "prob_left",
					  "alpha"],
		"bounds"   : [[0.05, 0.95],
					  [1.70, 8.00],
					  [0.70, 0.80],
					  [0.30, 0.45],
					  [2.00, 32.0]]
	}

	# generate Saltelli samples
	NUM_SAMPLES = 1024
	X = saltelli.sample(PROBLEM, NUM_SAMPLES, calc_second_order=False)
	L = int(X.shape[0]/128)
	X = X[idx*L:(idx+1)*L]

	start_time = time.time()

	# run each param combination
	for iter_idx, row in enumerate(X):

		# load graph corresponding to simga value of the current row
		init_w_scale = row[4]
		with open(f"../sda_graphs/{init_w_scale}.pickle", "rb") as f:
			_, communities, community_membership, initial_wealth = pickle.load(f)

		W, A, U, P, T, _, G = simulation (
			communities=communities,
			community_membership=community_membership,
			NUM_AGENTS=1250,
			STEPS=50,
			seed=SEED,
			PROJECT_COST=row[0],
			gain_right=row[1],
			ALPHA_BETA=row[2],
			prob_left=row[3],
			INIT_WEALTH_VALUES=initial_wealth,
			poisson_scale=row[5]
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