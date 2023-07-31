import sys
import time
import lzma
import pickle
from SALib.sample import saltelli
from network_model_cpt import *


if __name__ == "__main__":


	SEEDS = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]

	# parse arguments
	output_dir = sys.argv[1]
	idx = int(sys.argv[2]) - 1
	seed_idx = int(sys.argv[3])
	SEED = SEEDS[seed_idx]

	# load graph, and extract communities, community_membership
	with open("graph.pickle", "rb") as f:
		graph = pickle.load(f)
	communities = get_communities(graph)
	community_membership = get_community_membership(graph, communities)

	# problem definition
	PROBLEM = {
		"num_vars" : 6,
		"names"    : ["project_cost",
					  "gain_right",
					  "alpha_beta",
					  "prob_left",
					  "init_w_scale",
					  "poisson_scale"],
		"bounds"   : [[0.01, 2.00],
					  [1.70, 2.30],
					  [0.70, 0.80],
					  [0.30, 0.45],
					  [0.01, 0.15],
					  [8.00, 20.0]]
	}

	# generate Saltelli samples
	NUM_SAMPLES = 1024
	X = saltelli.sample(PROBLEM, NUM_SAMPLES, calc_second_order=False)
	L = int(X.shape[0]/128)
	X = X[idx*L:(idx+1)*L]

	start_time = time.time()

	# run each param combination
	for iter_idx, row in enumerate(X):

		W, A, U, P, T, _, G = simulation (
			graph=graph,
			communities=communities,
			community_membership=community_membership,
			NUM_AGENTS=1250,
			STEPS=50,
			seed=SEED,
			PROJECT_COST=row[0],
			gain_right=row[1],
			ALPHA_BETA=row[2],
			prob_left=row[3],
			init_wealth_scale=row[4],
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
		pickle.dump(data, lzma.open(output_dir + f"/{seed_idx}_{idx*L + iter_idx + 1}_cpt.pkl.lzma", 'wb'))

		print(f"JOB {idx} : finished seed {seed_idx}, param {idx*L + iter_idx + 1} at t = {(time.time() - start_time)/60:.0f} mins", 
			  flush=True)