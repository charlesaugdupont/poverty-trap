import sys
import pickle
from network_model import *
from SALib.sample import sobol

import time


SEED = 23


if __name__ == "__main__":

	# parse arguments
	output_dir = sys.argv[1]
	idx = int(sys.argv[2]) - 1

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
		"bounds"   : [[0.10, 3.00],
					  [1.70, 2.00],
					  [0.70, 0.80],
					  [0.30, 0.50],
					  [0.01, 0.15],
					  [1.00, 8.00],
					  [8.00, 20.00]]
	}

	# generate Saltelli samples
	NUM_SAMPLES = 1024
	X = sobol.sample(PROBLEM, NUM_SAMPLES, calc_second_order=False)
	L = int(X.shape[0]/128)
	X = X[idx*L:(idx+1)*L]

	start_time = time.time()

	# run experiments
	for iter_idx, row in enumerate(X):
		W, C, A, R, P, T, communities, G = simulation (
			NUM_AGENTS=1250, STEPS=50, seed=SEED, PROJECT_COST=row[0], gain_right=row[1], ALPHA_BETA=row[2], prob_left=row[3]
		)
		with open(output_dir + f"/{row[0]}_{row[1]}_{row[2]}_{row[3]}.pickle", "wb") as f:
			pickle.dump({
				"W":W.astype(np.float32),
				"C":C.astype(np.float32),
				"A":A.astype(np.float32),
				"R":R.astype(np.int8),
				"P":P,
				"T":np.array(list(T.values())).astype(np.int8),
				"communities":communities,
				"G":G.astype(np.float32)
			}, f)
	
	print(f"JOB {idx} : completed {L} runs in {(time.time() - start_time)/60:.3f} minutes.")