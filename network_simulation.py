import sys
from network_model import *
from SALib.sample import sobol


SEED = 1


if __name__ == "__main__":
	
	output_dir = sys.argv[1]
	idx = int(sys.argv[2])

	# define problem
	PROBLEM = {
		"num_vars" : 4,
		"names"    : ["project_cost", "default_a", "risky_return_left", "risky_return_right"],
		"bounds"   : [[0.01, 3.00],
					  [0.20, 1.20],
					  [1.30, 1.50],
					  [1.51, 1.70]]
	}
	# generate Saltelli samples
	NUM_SAMPLES = 256
	X = sobol.sample(PROBLEM, NUM_SAMPLES)[idx*16:(idx+1)*16]

	# perform experiments
	for row in X:
		W, I, communities, delta_pos, gamma_pos, success, allocations, C = simulation (
			NUM_AGENTS=1000, STEPS=50, seed=SEED, PROJECT_COST=row[0], DEFAULT_A=row[1], RL=row[2], RR=row[3]
		)
		with open(output_dir + f"/{row[0]}_{row[1]}_{row[2]}_{row[3]}.pickle", "wb") as f:
			pickle.dump({
				"W":W,
				"I":I,
				"C":C,
				"communities":communities,
				"delta_pos":delta_pos,
				"gamma_pos":gamma_pos,
				"success":success,
				"alloc":allocations,
			}, f)
