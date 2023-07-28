import numpy as np
import pickle
import lzma
import sys
import os


if __name__ == "__main__":

	output_dir = sys.argv[1]
	SEED = int(sys.argv[2]) - 1

	W_arrays = {f'{SEED}':{}}
	for f in os.listdir(f"../model_runs_cpt_random_{SEED}/"):
		data = pickle.load(lzma.open(f"../model_runs_cpt_random_{SEED}/{f}"))
		seed, param = f.split(".")[0].split("_")
		W_arrays[seed][param] = data["W"]

	for seed_idx, seed in enumerate(W_arrays):
		W_SEED = np.zeros((8192, 1250, 51), dtype=np.float16)
		param_W = W_arrays[seed]
		keys = list(param_W.keys())
		sorted_param_keys = sorted([int(x) for x in keys])
		for k_idx, k in enumerate(sorted_param_keys):
			W_SEED[k_idx] = param_W[str(k)].T
		pickle.dump(W_SEED, lzma.open(output_dir + f"/{SEED}_9216_1250_51_cpt_random.pkl.lzma", "wb"))