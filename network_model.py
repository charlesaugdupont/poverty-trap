import numpy as np
import networkx as nx
from scipy.stats import gengamma
from scipy.optimize import minimize

from cptopt.optimizer import MeanVarianceFrontierOptimizer
from cptopt.utility import CPTUtility

import pickle
import random


#################################################################################################

# UTILITY FUNCTIONS
def U(x, gamma=2.1, A=1.2):
	"""
	Utility function for consumption of agent.
	"""
	return A*x**(1-gamma) / (1-gamma)


def V(x, gamma=2.1):
	"""
	Utility function for bequest of agent to offspring.
	"""
	return x**(1-gamma) / (1-gamma)

#################################################################################################

# Network Construction

def build_graph(n, graph_type, graph_args):
	if graph_type == "powerlaw_cluster":
		G = nx.powerlaw_cluster_graph(n=n, **graph_args) 
	elif graph_type == "random_regular_graph":
		G = nx.random_regular_graph(n=n, **graph_args)
	assert nx.is_connected(G)
	return G


def get_adjacency(G):
	return dict((n, set(nbrdict.keys())) for n, nbrdict in G.adjacency())

#################################################################################################

# Communities

def get_communities(G):
	return list(nx.community.label_propagation_communities(G))
 

def get_node_community_map(communities):
	node_community_map = {}
	for i,c in enumerate(communities):
		for node in c:
			node_community_map[node] = i
	return node_community_map


def get_community_membership(G, communities):
	adjacency = get_adjacency(G)
	node_community_map = get_node_community_map(communities)
	membership = {i:set() for i in node_community_map}
	for i, neighbours in adjacency.items():
		for n in neighbours:
			membership[i].add(node_community_map[n])
		membership[i].add(node_community_map[i])
	membership = {k:np.array(list(v)) for k,v in membership.items()}
	return membership

#################################################################################################

# Gambles

def generate_gambles(N, left, right):
	"""
	Generate N gambles with 2 outcomes.
	"""
	probs     = np.random.uniform(0.30, 0.70, N)
	outcomes1 = np.random.uniform(0.90, 0.95, N)
	outcomes2 = np.random.uniform(left, right, N)

	gambles = []
	for i in range(N):
		gambles.append({
			"outcomes" : [outcomes1[i],outcomes2[i]],
			"probs"    : [probs[i], 1-probs[i]]
		})
	return gambles


def get_gamble_returns(P, size):
	return np.random.choice(P["outcomes"], p=P["probs"], size=size)

#################################################################################################

# Optimization

def utility(x, w, investment_returns, A, gamma):
	"""
	Compute expected utility.
	Args:
		x                   : consumption proportion
		w                   : wealth level of agent
		project_allocations : allocation of savings to risky projects
		project_returns     : expected project return
		A                   : utility function parameter
		gamma               : utility function parameter
	Returns:
		utility
	"""
	# utility from consumption
	consumption_utility = U(w*x, A=A, gamma=gamma)

	# expected utility from projects
	project_utility = V(sum(w*(1-x)*investment_returns))
	
	return - (consumption_utility + project_utility)


#################################################################################################

# Simulation

def simulation(NUM_AGENTS=500, STEPS=50, SAFE_RETURN=1.10, DEFAULT_A=1.2, DEFAULT_GAMMA=2.1,
			   PROJECT_COST=3.0, W0=0.8, W1=1.2, DELTA_POS_L=0.5, DELTA_POS_R=0.78, graph=None,
			   NUM_GAMBLE_SAMPLES=1000, graph_type="powerlaw_cluster", graph_args={"m":2, "p":0.5},
			   RL=1.2, RR=1.5, GAMMA_POS_L=3, GAMMA_POS_R=9, seed=None, use_data=False):
	"""
	Runs ABM model.
	Args:
		NUM_AGENTS    	   : number of agents
		STEPS         	   : number of steps
		SAFE_RETURN   	   : safe return coefficient (> 1.0)
		DEFAULT_A     	   : parameter used in utility functions
		DEFAULT_GAMMA 	   : parameter used in utility functions
		PROJECT_COST  	   : minimum cost for project to be undertaken
		W0            	   : left bound for uniform random wealth initialization
		W1            	   : right bound for uniform random wealth initialization
		GAMMA_POS_L	  	   : left bound for uniform random risk initialization
		GAMMA_POS_R		   : right bound for uniform random risk initialization
		NUM_GAMBLE_SAMPLES : number of random samples for cumulative prospect theory utility
		graph 		  	   : NetworkX graph
		graph_type    	   : type of graph to use
		graph_args    	   : arguments for graph construction, specific to graph type passed
	Returns:
		WEALTH      : (STEPS, NUM_AGENTS) array containing wealth levels of agents at each iteration
		communities : dict from community ID to list of members
	"""
	if seed:
		random.seed(seed)
		np.random.seed(seed)

	# construct graph and adjacency matrix
	G = graph or build_graph(NUM_AGENTS, graph_type, graph_args)

	# extract communities
	communities = get_communities(G)
	print(f"{len(communities)} communities.")

	# get community membership of each agent
	community_membership = get_community_membership(G, communities)
	communities = {c:[] for c in range(len(communities))}
	for i, comms in community_membership.items():
		for c in comms:
			communities[c].append(i)

	# global attributes
	GAMBLES = generate_gambles(len(communities), left=RL, right=RR)
	GAMBLE_SAMPLES = np.zeros((NUM_GAMBLE_SAMPLES, len(GAMBLES)))
	for i,g in enumerate(GAMBLES):
		GAMBLE_SAMPLES[:,i] = np.random.choice(g["outcomes"], NUM_GAMBLE_SAMPLES, p=g["probs"]) - 1
	GAMBLE_RETURNS = np.row_stack([[get_gamble_returns(P, size=STEPS) for P in GAMBLES]])
	gamble_success = np.zeros((len(GAMBLES)))
	gamble_averages = np.mean(GAMBLE_SAMPLES, axis=0)

	# agent attributes
	C      = np.zeros((STEPS, NUM_AGENTS))
	INCOME = np.zeros((STEPS, NUM_AGENTS))
	WEALTH = np.random.uniform(W0, W1, size=(STEPS+1, NUM_AGENTS))
	ATTENTION = np.random.uniform(0.2, 0.8, size=NUM_AGENTS)
	DELTA_POS = np.random.uniform(DELTA_POS_L, DELTA_POS_R, size=NUM_AGENTS)
	GAMMA_POS = np.random.uniform(GAMMA_POS_L, GAMMA_POS_R, size=NUM_AGENTS)
	UTILITIES = [CPTUtility(gamma_pos=GAMMA_POS[i], gamma_neg=11.4, delta_pos=DELTA_POS[i], delta_neg=0.79) for i in range(NUM_AGENTS)]
	AGENT_EXPECTED_RETURNS = [np.concatenate([gamble_averages[community_membership[i]]+1, [SAFE_RETURN]]) for i in range(NUM_AGENTS)]
	ALLOC = []

	if use_data:
		print("Loading pre-computed optimal portfolios...")
		with open("cpt_data.pickle", "rb") as f:
			data = pickle.load(f)
		ALLOC = data["alloc"]
		AGENT_EXPECTED_RETURNS = data["agent_expected_returns"]
		DELTA_POS = data["delta_pos"]
	else:
		# compute optimal portfolios for agents
		print("Computing optimal portfolios...")
		for i in range(NUM_AGENTS):
			mv = MeanVarianceFrontierOptimizer(UTILITIES[i])
			samples = GAMBLE_SAMPLES[:,community_membership[i]]
			samples_with_safe_gamble = np.column_stack([samples, np.repeat(SAFE_RETURN-1, samples.shape[0])])
			mv.optimize(samples_with_safe_gamble)
			ALLOC.append(mv.weights)

	# simulation
	print("Performing time stepping...")
	for step in range(STEPS):

		project_contributions = np.zeros((len(GAMBLES)))

		# serial approach (better than multiprocessing for O(10^3) agents, but worse for >= O(10^4))
		for i in range(NUM_AGENTS):
			C[step][i] = minimize(utility, x0=0.5, bounds=[(0.05, 0.95)], method='SLSQP',
							args=(WEALTH[step][i],
								  AGENT_EXPECTED_RETURNS[i],
								  DEFAULT_A,
								  DEFAULT_GAMMA)).x[0]
			project_contributions[community_membership[i]] += WEALTH[step][i]*(1-C[step][i])*ALLOC[i][:-1]

		# get gamble returns
		returns = (project_contributions >= PROJECT_COST) * GAMBLE_RETURNS[:,step]

		# update agent wealth and income
		for i in range(NUM_AGENTS):
			# investment proportion
			I = 1-C[step][i]

			# income from investments = sum of risky investment returns + safe investment return
			risky_return = sum(WEALTH[step][i]*I*ALLOC[i][:-1]*(returns[community_membership[i]]))
			safe_return  = WEALTH[step][i]*I*ALLOC[i][-1]*(SAFE_RETURN)
			INCOME[step][i] = risky_return + safe_return
			
			# new wealth = current wealth - consumption + income from investments
			WEALTH[step+1][i] = WEALTH[step][i] * (1-C[step][i]) + INCOME[step][i]

	return WEALTH, INCOME, communities, DELTA_POS, GAMMA_POS, gamble_success, ALLOC, C

#################################################################################################

# Utilities and Metrics

def count_crossover_points(W, communities=None):
	"""
	Count number of crossover points at agent or community level.
	Args:
		W           : (STEPS, NUM_AGENTS) array
		communities : dict from community ID to list of members (community level if this is provided)
	Returns:
		number of crossover points.
	"""
	# communities is specified, so count crossover points at community-level
	if communities:
		crossover_points = {c:0 for c in range(len(communities))}
		for c, agent_list in communities.items():
			trajectory = np.mean(W[:,agent_list], axis=1)
			for i in range(len(trajectory)-2):
				if (trajectory[i+1] - trajectory[i] > 0 and trajectory[i+2] - trajectory[i+1] < 0) or \
				   (trajectory[i+1] - trajectory[i] < 0 and trajectory[i+2] - trajectory[i+1] > 0):
						crossover_points[c] += 1

	# otherwise count at agent level
	else:
		num_agents = W.shape[1]
		crossover_points = {a:0 for a in range(num_agents)}
		for a in range(num_agents):
			trajectory = W[:,a]
			for i in range(len(trajectory)-2):
				if (trajectory[i+1] - trajectory[i] > 0 and trajectory[i+2] - trajectory[i+1] < 0) or \
				   (trajectory[i+1] - trajectory[i] < 0 and trajectory[i+2] - trajectory[i+1] > 0):
						crossover_points[a] += 1

	return crossover_points


def get_community_income(I, communities):
	return [np.mean(I[-1][communities[c]]) for c in communities]


def get_community_wealth(W, communities):
	return [np.mean(W[-1][communities[c]]) for c in communities]