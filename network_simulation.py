import sys
import time
import lzma
import pickle
from network_model import *
from SALib.sample import saltelli


if __name__ == "__main__":

	# parse arguments
	output_dir = sys.argv[1]
	idx = int(sys.argv[2]) - 1

	# load graph, and extract communities, community_membership
	with open("graph.pickle", "rb") as f:
		graph = pickle.load(f)
	communities = get_communities(graph)
	community_membership = get_community_membership(graph, communities)

	# problem definition
	PROBLEM = {
		"num_vars" : 7,
		"names"    : ["project_cost",
					  "gain_right",
					  "alpha_beta",
					  "prob_left",
					  "init_w_scale",
					  "risk_scale",
					  "poisson_scale"],
		"bounds"   : [[0.01, 2.00],
					  [1.70, 2.30],
					  [0.70, 0.80],
					  [0.30, 0.45],
					  [0.01, 0.15],
					  [5.00, 20.0],
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

		# 10 repetitions to capture stochastic effects
		for rep_idx, SEED in enumerate([1, 2, 3, 5, 8, 13, 21, 34, 55, 89]):

			W, A, R, P, T, _, G = simulation (
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
			   	risk_scale=row[5],
			   	poisson_scale=row[6]
			)

			# store results
			data = {
				"W":W,
				"A":A,
				"R":R,
				"P":P,
				"T":np.array(list(T.values())).astype(np.int16),
				"G":G,
				"params":tuple(row),
			}
			pickle.dump(data, lzma.open(output_dir + f"/{rep_idx}_{idx*L + iter_idx + 1}.pkl.lzma", 'wb'))

		print(f"JOB {idx} : finished param {iter_idx+1} at t = {(time.time() - start_time)/60:.0f} mins")