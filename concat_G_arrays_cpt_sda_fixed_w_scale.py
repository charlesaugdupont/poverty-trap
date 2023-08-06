import numpy as np
import pickle
import lzma
import sys
import os


if __name__ == "__main__":

	output_dir = sys.argv[1]
	SEED = int(sys.argv[2]) - 1

	G_arrays = {f'{SEED}':{}}
	for f in os.listdir(f"../model_runs_cpt_sda_fixed_w_scale_{SEED}/"):
		data = pickle.load(lzma.open(f"../model_runs_cpt_sda_fixed_w_scale_{SEED}/{f}"))
		seed, param = f.split(".")[0].split("_")[:2]
		G_arrays[seed][param] = data["G"].T

	for seed_idx, seed in enumerate(G_arrays):
		G_SEED = []
		param_G = G_arrays[seed]
		keys = list(param_G.keys())
		sorted_param_keys = sorted([int(x) for x in keys])
		for k_idx, k in enumerate(sorted_param_keys):
			G_SEED.append(param_G[str(k)])
		pickle.dump(G_SEED, lzma.open(output_dir + f"/G_{SEED}_cpt_sda_fixed_w_scale.pkl.lzma", "wb"))