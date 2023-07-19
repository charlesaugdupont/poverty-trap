import numpy as np
import contextlib
import pickle
import time
import lzma
import sys

from gldpy import GLD
gld = GLD('FMKL')


if __name__ == "__main__":

	start_time = time.time()

	output_dir = sys.argv[1]
	TASK_IDX = int(sys.argv[2]) - 1

	with open("augmented_communities.pickle", "rb") as f:
		communities = pickle.load(f)

	# construct mapping from task_idx to communities
	mapping = {i:[] for i in range(128)}
	for i, key in enumerate(communities):
		mapping[i%128].append(key)
	mapping = {k:np.array(v) for k,v in mapping.items()}

	# determine which communities current task is responsible for
	target_communities = mapping[TASK_IDX]

	# go through all seed results
	for seed in range(10):
		lambda_fit_results = {c:{} for c in target_communities}
		results = pickle.load(lzma.open(f"../concat_W_arrays/{seed}_9216_1250_51.pkl.lzma"))
		n_param_combos = results.shape[0]
		n_steps = results.shape[2]
		n_lambdas = 4

		# perform fits for each target community
		for c in target_communities:
			lambdas_LMM = np.zeros((n_param_combos, n_steps, n_lambdas))
			for param_id in range(n_param_combos):
				for step in range(n_steps):
					data = results[param_id][communities[c]][step]
					with contextlib.redirect_stdout(None):
						lambdas_LMM[param_id][step] = gld.fit_LMM(data, [0.5,0.5], disp_fit=False, maxiter=1000, maxfun=1000)
			lambda_fit_results[c] = lambdas_LMM

		with open(output_dir + f"/{TASK_IDX}_{seed}_LLM_communities.pickle", "wb") as f:
			pickle.dump(lambdas_LMM, f)