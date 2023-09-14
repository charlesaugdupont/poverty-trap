import numpy as np
import networkx as nx


#################################################################################################

# Network Construction

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