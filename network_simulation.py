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
		"num_vars" : 3,
		"names"    : ["project_cost", "rr", "alpha_beta"],
		"bounds"   : [[0.10, 3.00],
					  [1.70, 2.00],
					  [0.70, 0.80]]
	}

	# generate Saltelli samples
	NUM_SAMPLES = 256
	X = sobol.sample(PROBLEM, NUM_SAMPLES)
	L = int(X.shape[0]/128)
	X = X[idx*L:(idx+1)*L]

	start_time = time.time()

	# run experiments
	for iter_idx, row in enumerate(X):
		W, C, A, R, P, T, communities, G = simulation (
			NUM_AGENTS=1250, STEPS=50, seed=SEED, PROJECT_COST=row[0], RR=row[1], ALPHA_BETA=row[2]
		)
		with open(output_dir + f"/{row[0]}_{row[1]}_{row[2]}.pickle", "wb") as f:
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