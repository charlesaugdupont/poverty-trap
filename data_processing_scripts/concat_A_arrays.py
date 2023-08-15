import pickle
import lzma
import sys
import os


if __name__ == "__main__":

	output_dir = sys.argv[1]
	
	multi_seed = [
		"model_runs",
		"model_runs_random",
		"model_runs_risk_aversion_experiment",
		"model_runs_sda"
	]

	single_seed = [f"model_runs_cpt_{i}" for i in range(10)] + \
				  [f"model_runs_cpt_random_{i}" for i in range(10)] + \
				  [f"model_runs_cpt_sda_{i}" for i in range(10)]
	

	RESULTS = {}

	for path in multi_seed:
		RESULTS[path] = {}
		files = os.listdir(f"../{path}")
		for i in range(10):
			first_seed_file = [f for f in files if f[0] == str(i)][0]
			data = pickle.load(lzma.open(f"../{path}/{first_seed_file}"))
			RESULTS[path][str(i)] = data["A"]

	for path in single_seed:
		modified_path = path[:-2]
		i = path[-1]
		if modified_path not in RESULTS:
			RESULTS[modified_path] = {}
		file = os.listdir(f"../{path}")[0]
		data = pickle.load(lzma.open(f"../{path}/{file}"))
		RESULTS[modified_path][i] = data["A"]	

	with open(output_dir+"/attention_arrays.pickle", "wb") as f:
		pickle.dump(RESULTS, f)