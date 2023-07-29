import sys
import pickle
from SALib.sample import sobol
from network_model_sda import *


def sda_prob(d, a, b):
    return 1 / (1 + ((1/b)*d)**a)


def sda_graph(P):
    G = nx.Graph()
    # create nodes
    for i in range(1250):
        G.add_node(i)
    coin_flips = np.random.uniform(size=780625) # 780625 = number of pairwise distances for 1250 agents
    k = 0
    # add edges probabilistically
    for i in range(len(P)):
        for j in range(i+1, len(P)):
            if P[i][j] > coin_flips[k]:
                G.add_edge(i,j)
            k += 1  
    return G


if __name__ == "__main__":

	SEEDS = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]

	# parse arguments
	output_dir = sys.argv[1]
	idx = int(sys.argv[2]) - 1
	
	# set random seed
	SEED = SEEDS[idx]
	random.seed(SEED)
	np.random.seed(SEED)

	# problem definition
	PROBLEM = {
		"num_vars" : 7,
		"names"    : ["project_cost",
					  "gain_right",
					  "alpha_beta",
					  "prob_left",
					  "init_w_scale",
					  "risk_scale",
					  "poisson_scale"],
		"bounds"   : [[0.01, 2.00],
					  [1.70, 2.30],
					  [0.70, 0.80],
					  [0.30, 0.45],
					  [0.01, 0.15],
					  [5.00, 20.0],
					  [8.00, 20.0]]
	}

	# generate Saltelli samples
	NUM_SAMPLES = 1024
	X = sobol.sample(PROBLEM, NUM_SAMPLES, calc_second_order=False)

	unique_w_scales = set(X[:,4])
	GRAPHS = {}
	print(f"Starting graph generation for seed {SEED}...", flush=True)
	for scale in unique_w_scales:

		# create random initial wealth distribution
		W = np.random.normal(1, scale, size=1250)

		# compute pairwise distances
		pairwise_distances = []
		for i in range(len(W)):
			for j in range(i+1, len(W)):
				pairwise_distances.append(np.abs(W[i]-W[j]))
		mean_dist = np.mean(pairwise_distances)

		# SDA graph parameters
		a = 16
		b = mean_dist/15
		prob_attach = np.zeros((1250,1250))
		k = 0
		for i in range(len(W)):
			for j in range(i+1, len(W)):
				p = sda_prob(pairwise_distances[k], a=a, b=b)
				prob_attach[i][j] = p
				prob_attach[j][i] = p
				k+=1
		G = sda_graph(prob_attach)

		# ensure that graph is connected
		connected_components = list(nx.connected_components(G))
		largest_component = list(connected_components[0])
		largest_component_wealths = W[np.array(largest_component)]
		for i in range(1, len(connected_components)):
			comp = list(connected_components[i])
			node = np.random.choice(comp)
			closest_largest_component_node = (np.abs(largest_component_wealths - W[node])).argmin()
			G.add_edge(node, largest_component[closest_largest_component_node])
		assert nx.is_connected(G)
			
		# extract communities and construct community membership dictionary
		communities = get_communities(G)
		community_membership = get_community_membership(G, communities)
		GRAPHS[scale] = (G, communities, community_membership, W)

	with open(output_dir + f"/sda_graphs_{SEED}.pkl.lzma", 'wb') as f:
		pickle.dump(GRAPHS, f)