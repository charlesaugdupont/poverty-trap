import numpy as np
import pickle
import lzma
import time
import sys
import os


if __name__ == "__main__":

	output_dir = sys.argv[1]

	start_time = time.time()

	print(f"{time.time() - start_time} : Starting data loding...")

	G_arrays = {f'{i}':{} for i in range(10)}
	for f in os.listdir("../model_runs_sda/"):
		data = pickle.load(lzma.open(f"../model_runs_sda/{f}"))
		seed, param = f.split(".")[0].split("_")[:2]
		G_arrays[seed][param] = data["G"].T
	print(f"{time.time()-start_time} : Finished loading data.", flush=True)

	for seed_idx, seed in enumerate(G_arrays):
		G_SEED = []
		param_G = G_arrays[seed]
		keys = list(param_G.keys())
		sorted_param_keys = sorted([int(x) for x in keys])
		for k_idx, k in enumerate(sorted_param_keys):
			G_SEED.append(param_G[str(k)])
		pickle.dump(G_SEED, lzma.open(output_dir + f"/G_{seed}_sda.pkl.lzma", "wb"))
		print(f"{time.time()-start_time} : Finished concatenating and storing seed {seed}", flush=True)