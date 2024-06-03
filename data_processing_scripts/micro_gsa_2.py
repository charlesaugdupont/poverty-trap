from GDMaps_PCE_GSA import *
import pickle
import lzma


def run_GDMaps(p, 
			   data, 
			   num_runs=10, 
			   n_keep=3, 
			   parsim=True):
	
	evals_diff_runs  = []
	evecs_diff_runs  = []
	coord_diff_runs = []
	g_diff_runs = []
	residuals_diff_runs = []
	index_diff_runs = []
	
	for i in range(num_runs):
		data_all = data[i]
		if parsim:
			g, coord, _, residuals, index, evals, evecs = GDMaps(data=data_all, 
																 n_evecs=20,
																 n_keep=n_keep,
															  	 parsim=parsim,
																 p=p).get()
			evals_diff_runs.append(evals)
			evecs_diff_runs.append(evecs)
			coord_diff_runs.append(coord)
			g_diff_runs.append(g)
			residuals_diff_runs.append(residuals)
			index_diff_runs.append(index)
		else:
			g, coord, _, evals, evecs = GDMaps(data=data_all, 
											   n_evecs=20,
											   n_keep=n_keep,
											   parsim=parsim,
											   p=p).get()
			evals_diff_runs.append(evals)
			evecs_diff_runs.append(evecs)
			coord_diff_runs.append(coord)
			g_diff_runs.append(g)
			
	return (evals_diff_runs, evecs_diff_runs, coord_diff_runs, g_diff_runs,
			residuals_diff_runs, index_diff_runs)


if __name__ == "__main__":
	SEEDS = [8,9,10,11]
	P = 91
	for seed in SEEDS:
		data = pickle.load(lzma.open(f"{seed}_paper.pkl.lzma"))
		data = data[:,:,:100].reshape(1, 7168, 350, 350)
		results = run_GDMaps(p=P, data=data, num_runs=1)
		with open(f"micro_{seed}.pickle", "wb") as f:
			pickle.dump(results, f)