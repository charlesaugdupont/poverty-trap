from gldpy import GLD
import numpy as np
import contextlib
import pickle
import lzma
import sys


if __name__ == "__main__":

	output_dir = sys.argv[1]
	seed = int(sys.argv[2]) - 1

	# load results for specified seed
	results = pickle.load(lzma.open("../concat_W_arrays/{seed}_9216_1250_51.pkl.lzma"))

	n_param_combos = results.shape[0]
	n_steps = results.shape[2]
	n_lambdas = 4

	lambdas_LMM = np.zeros((n_param_combos, n_steps, n_lambdas))
	gld = GLD('FMKL')
	for param_id in range(n_param_combos):
		for step in range(n_steps):
			data = results[param_id][:,step]
			with contextlib.redirect_stdout(None):
				lambdas_LMM[param_id][step] = gld.fit_LMM(data, [0.5,0.5], disp_fit=False, maxiter=1000, maxfun=1000)

	with open(output_dir + f"/{seed}_LLM_results.pickle", "wb") as f:
		pickle.dump(lambdas_LMM, f)