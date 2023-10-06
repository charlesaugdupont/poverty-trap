import numpy as np
import pickle
import lzma
import sys
import os


if __name__ == "__main__":

	output_dir = sys.argv[1]
	SEED = int(sys.argv[2]) - 1

	W_arrays = {f'{SEED}':{}}
	for f in os.listdir(f"./new_model_runs_paper_{SEED}/"):
		data = pickle.load(lzma.open(f"./new_model_runs_paper_{SEED}/{f}"))
		seed, param = f.split(".")[0].split("_")[:2]
		W_arrays[seed][param] = data["W"]

	for seed_idx, seed in enumerate(W_arrays):
		W_SEED = np.zeros((7168, 1225, 101), dtype=np.float16)
		param_W = W_arrays[seed]
		keys = list(param_W.keys())
		sorted_param_keys = sorted([int(x) for x in keys])
		for k_idx, k in enumerate(sorted_param_keys):
			W_SEED[k_idx] = param_W[str(k)].T
		pickle.dump(W_SEED, lzma.open(output_dir + f"/{SEED}_paper.pkl.lzma", "wb"))