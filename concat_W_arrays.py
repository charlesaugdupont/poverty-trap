import numpy as np
import pickle
import lzma
import sys
import os


if __name__ == "__main__":

	output_dir = sys.argv[1]

	W_arrays = {f'{i}':{} for i in range(10)}
	for f in os.listdir("../output_directory/"):
		data = pickle.load(lzma.open(f"../output_directory/{f}"))
		seed, param = f.split(".")[0].split("_")
		W_arrays[seed][param] = data["W"]

	W_ALL = np.zeros((10, 9216, 1250, 51), dtype=np.float16)
	for seed_idx, seed in enumerate(W_arrays):
		param_W = W_arrays[seed]
		keys = list(param_W.keys())
		sorted_param_keys = sorted([int(x) for x in keys])
		for k_idx, k in enumerate(sorted_param_keys):
			W_ALL[seed_idx][k_idx] = param_W[str(k)].T

	for seed in range(W_ALL.shape[0]):
		pickle.dump(W_ALL[seed], lzma.open(output_dir + f"/{seed}_9216_1250_51.pkl.lzma", "wb"))