import numpy as np
import pickle
import lzma
import time
import sys
import os


if __name__ == "__main__":

	output_dir = sys.argv[1]

	start_time = time.time()

	print(f"{time.time() - start_time} : Starting data loading...")

	W_arrays = {f'{i}':{} for i in range(10)}
	for f in os.listdir("../model_runs/"):
		data = pickle.load(lzma.open(f"../model_runs/{f}"))
		seed, param = f.split(".")[0].split("_")
		W_arrays[seed][param] = data["W"]
	print(f"{time.time()-start_time} : Finished loading data.", flush=True)

	for seed_idx, seed in enumerate(W_arrays):
		W_SEED = np.zeros((9216, 1250, 51), dtype=np.float16)
		param_W = W_arrays[seed]
		keys = list(param_W.keys())
		sorted_param_keys = sorted([int(x) for x in keys])
		for k_idx, k in enumerate(sorted_param_keys):
			W_SEED[k_idx] = param_W[str(k)].T
		pickle.dump(W_SEED, lzma.open(output_dir + f"/{seed}_9216_1250_51.pkl.lzma", "wb"))
		print(f"{time.time()-start_time} : Finished concatenating and storing seed {seed}", flush=True)