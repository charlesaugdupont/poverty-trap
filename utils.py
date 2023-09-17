import numpy as np
import networkx as nx


#################################################################################################

# Network

def construct_sda_graph(W, alpha, b=1/15):
	"""
	Constructs a Social Distance Attachment graph.
	Args:
		W 	  : array of agent wealth values
		alpha : homophily parameter
		b     : characteristic distance factor
	"""
	# get number of agents
	N = len(W)

	# compute pairwise distances
	D = []
	for i in range(N):
		for j in range(i+1, N):
			D.append(np.abs(W[i]-W[j]))

	# compute characteristic distance
	mean_dist = np.mean(D)
	characteristic_distance = mean_dist * b

	# initialize graph with one node per agent
	G = nx.Graph()
	for i in range(N):
		G.add_node(i)

	# add edges to graph probabilistically
	coin_flips = np.random.uniform(size=int(N*(N-1)/2)) # total number of pairwise distances
	k = 0
	for i in range(N):
		for j in range(i+1, N):
			# attachment probability is given by Fermi-Dirac distribution
			p = 1/(1+((1/characteristic_distance)*D[k])**alpha)
			if p > coin_flips[k]:
				G.add_edge(i,j)
			k += 1

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

	return G


def get_adjacency(G):
	"""
	Constructs adjacency dictionary.
	"""
	return dict((n, set(nbrdict.keys())) for n, nbrdict in G.adjacency())

#################################################################################################

# Communities

def get_communities(G):
	"""
	Extracts communities from networkx graph using label propagation.
	"""
	return list(nx.community.label_propagation_communities(G))
 

def get_node_community_map(communities):
	"""
	Constructs a mapping from each node to the community index it is part of.
	Args:
		communities : list of communities arising from label propagation algorithm
	"""
	node_community_map = {}
	for i,c in enumerate(communities):
		for node in c:
			node_community_map[node] = i
	return node_community_map


def get_community_membership(G, communities):
	"""
	Constructs a mapping from each node to multiple community indices
	(including the one it is a part of, and the ones its neighbors are part of).
	Args:
		G 			: networkx graph
		communities : list of communities arising from label propagation algorithm
	"""
	L = len(communities)
	adjacency = get_adjacency(G)
	node_community_map = get_node_community_map(communities)
	membership = {i:{node_community_map[i], L} for i in node_community_map}
	for i, neighbours in adjacency.items():
		for n in neighbours:
			membership[i].add(node_community_map[n])
	membership = {k:np.array(list(v)) for k,v in membership.items()}
	return membership


def get_augmented_communities(community_membership):
	"""
	Construct mapping from each community index to agent indices belong to the community.
	Args:
		community_membership : mapping from each agent to multiple community indices
	"""
	augmented_communities = {}
	for agent in community_membership:
		comms = community_membership[agent]
		for c in comms:
			if c not in augmented_communities:
				augmented_communities[c] = {agent}
			else:
				augmented_communities[c].add(agent) 
	augmented_communities = {k:np.array(list(v)) for k,v in augmented_communities.items()}
	del augmented_communities[max(augmented_communities.keys())]


def get_community_project_costs(W, augmented_communities, theta):
	"""
	Compute the project investment threshold for each community.
	Args:
		W 					  : array of agent wealths
		augmented_communities : mapping from community index to agent indices
		theta        		  : scalar between 0 and 1
	"""
	# size is number of communites + 1 due to "safe project"
	project_costs = np.zeros((len(augmented_communities)+1,))
	for i in range(len(augmented_communities)):
		project_costs[i] = np.sum(W[augmented_communities[i]])*theta
	return project_costs

#################################################################################################

# Gambles

def generate_gambles(N, gain_right_bound, prob_left):
	"""
	Generate N gambles with 2 outcomes each.
	"""
	loss_probs = np.random.uniform(prob_left, 1-prob_left, N)
	gain_probs = 1 - loss_probs
	outcomes1 = np.random.uniform(0.90, 0.95, N)
	outcomes2 = np.random.uniform(1.6, gain_right_bound, N)

	gambles = []
	for i in range(N):
		gambles.append({
			"outcomes" : [outcomes1[i], outcomes2[i]],
			"probs"    : [loss_probs[i], gain_probs[i]]
		})
	return gambles


def get_gamble_returns(P, size):
	"""
	Generates some random gamble returns based on their outcomes and branch probabilities.
	"""
	return np.random.choice(P["outcomes"], p=P["probs"], size=size)