import argparse
import numpy as np
from tqdm import tqdm
import networkx as nx
from model_banerjee import U,V
from multiprocessing import Pool
from scipy.stats import gengamma
from scipy.optimize import minimize

from cptopt.optimizer import MeanVarianceFrontierOptimizer
from cptopt.utility import CPTUtility

import pickle
import random


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

def generate_gambles(N):
	"""
	Generate N gambles with 2 outcomes.
	"""
	probs     = np.random.uniform(0.30, 0.70, N)
	outcomes1 = np.random.uniform(0.3, 1.0, N)
	outcomes2 = np.random.uniform(1.5, 1.7, N)

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


def optimize_utility(arg_tuple):
	# unpack tuple
	wealth, allocation, gamble_averages, risk, DEFAULT_A, DEFAULT_GAMMA, SAFE_RETURN = arg_tuple

	# run minimization
	res = minimize(utility, x0=0.5, bounds=[(0.05, 0.95)], method='SLSQP', 
				   args=(wealth, allocation, gamble_averages, risk, DEFAULT_A, DEFAULT_GAMMA, SAFE_RETURN))
	return res.x[0]

#################################################################################################

# Simulation

def simulation(NUM_AGENTS=500, STEPS=50, SAFE_RETURN=1.10, DEFAULT_A=1.2, DEFAULT_GAMMA=2.1,
			   PROJECT_COST=3.0, W0=0.8, W1=1.2, GAMMA_POS_L=3, GAMMA_POS_R=9, graph=None,
			   NUM_GAMBLE_SAMPLES=1000, graph_type="powerlaw_cluster", graph_args={"m":2, "p":0.5},
			   seed=None):
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

	multiprocess = NUM_AGENTS >= 10000

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
	GAMBLES = generate_gambles(len(communities))
	GAMBLE_SAMPLES = np.zeros((NUM_GAMBLE_SAMPLES, len(GAMBLES)))
	for i,g in enumerate(GAMBLES):
		GAMBLE_SAMPLES[:,i] = np.random.choice(g["outcomes"], NUM_GAMBLE_SAMPLES, p=g["probs"]) - 1
	GAMBLE_RETURNS = np.row_stack([[get_gamble_returns(P, size=STEPS) for P in GAMBLES]])
	gamble_success = np.zeros((len(GAMBLES)))
	gamble_averages = np.mean(GAMBLE_SAMPLES, axis=0)

	# agent attributes
	C      = np.zeros((NUM_AGENTS))
	INCOME = np.zeros((STEPS, NUM_AGENTS))
	WEALTH = np.random.uniform(W0, W1, (STEPS+1, NUM_AGENTS))
	GAMMA_POS = np.random.uniform(GAMMA_POS_L, GAMMA_POS_R, NUM_AGENTS)
	UTILITIES = [CPTUtility(gamma_pos=GAMMA_POS[i], gamma_neg=10, delta_pos=0.77, delta_neg=0.79) for i in range(NUM_AGENTS)]
	AGENT_GAMBLE_AVERAGES = [np.concatenate([gamble_averages[community_membership[i]], [SAFE_RETURN]]) for i in range(NUM_AGENTS)]
	ALLOC = []
	AGENT_EXPECTED_RETURNS = []
	# compute optimal portfolios for agents
	print("Computing optimal portfolios...")
	for i in tqdm(range(NUM_AGENTS)):
		mv = MeanVarianceFrontierOptimizer(UTILITIES[i])
		samples = GAMBLE_SAMPLES[:,community_membership[i]]
		samples_with_safe_gamble = np.column_stack([samples, np.repeat(SAFE_RETURN-1, samples.shape[0])])
		mv.optimize(samples_with_safe_gamble)
		ALLOC.append(mv.weights)
		# pre-compute expected returns - note: if the mean variance optimizer is re-run, this needs to be updated!
		AGENT_EXPECTED_RETURNS.append(ALLOC[-1]*AGENT_GAMBLE_AVERAGES[i])
	with open("cpt_data.pickle", "wb") as f:
		pickle.dump({"alloc":ALLOC, 
					 "agent_expected_returns":AGENT_EXPECTED_RETURNS,
					 "gamma_pos":GAMMA_POS}, f)

	# simulation
	print("Performing time stepping...")
	for step in tqdm(range(STEPS)):

		project_contributions = np.zeros((len(GAMBLES)))

		# all agents perform optimization step and we sum up project contributions
		# if multiprocess:
		# 	args = [(WEALTH[step][i], ALLOC[i], 
		# 			gamble_averages[community_membership[i]],
		# 			RISK[i], DEFAULT_A, DEFAULT_GAMMA, SAFE_RETURN)
		# 			for i in range(NUM_AGENTS)]
		# 	with Pool() as pool:
		# 		results = pool.map(optimize_utility, args)
		# 	C = np.array(results)
		# 	for i in range(NUM_AGENTS):
		# 		project_contributions[community_membership[i]] += \
		# 							np.sum(WEALTH[step][i]*ALLOC[i]*RISK[i]*(1-C[i]))  

		# serial approach (better than multiprocessing for O(10^3) agents, but worse for >= O(10^4))
		for i in range(NUM_AGENTS):
			C[i] = minimize(utility, x0=0.5, bounds=[(0.05, 0.95)], method='SLSQP',
							args=(WEALTH[step][i],
								  AGENT_EXPECTED_RETURNS[i],
								  DEFAULT_A,
								  DEFAULT_GAMMA)).x[0]
			project_contributions[community_membership[i]] += WEALTH[step][i]*(1-C[i])*ALLOC[i][:-1]

		# run projects
		returns = np.zeros((len(GAMBLES)))
		for idx in range(len(GAMBLES)):
			if project_contributions[idx] >= PROJECT_COST:
				returns[idx] = GAMBLE_RETURNS[idx][step]
				gamble_success[idx] += 1

		for i in range(NUM_AGENTS):
			I = 1-C[i]

			# approach 1?
			invested_wealth = I*WEALTH[step][i]
			safe_return  = invested_wealth * ALLOC[i][-1] * SAFE_RETURN
			risky_return = sum(invested_wealth * ALLOC[i][:-1] * returns[community_membership[i]])
			INCOME[step][i] = safe_return + risky_return
			WEALTH[step+1][i] = WEALTH[step][i] - WEALTH[step][i]*C[i] + INCOME[step][i]

			# approach 2?
			INCOME[step][i] = sum(WEALTH[step][i]*I*ALLOC[i][:-1]*(returns[community_membership[i]]-1)) + \
							  WEALTH[step][i]*I*ALLOC[i][-1]*SAFE_RETURN
			WEALTH[step+1][i] = WEALTH[step][i] - WEALTH[step][i]*C[i] + INCOME[step][i]

	return WEALTH, INCOME, communities, GAMMA_POS, gamble_success, ALLOC, C

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


def fit_generalized_gamma(data):
	params = gengamma.fit(data)
	return params

#################################################################################################

if __name__ == "__main__":

	# parse command-line arguments
	parser = argparse.ArgumentParser(description="Run ABM simulation.")
	parser.add_argument("--agents", help="Number of agents", default=1000, type=int)
	parser.add_argument("--steps", help="Number of steps", default=100, type=int)
	parser.add_argument("--project-cost", help="Cost of starting a project", default=0.5, type=float)
	args = parser.parse_args()

	NUM_AGENTS   = args.agents
	STEPS	  	 = args.steps
	PROJECT_COST = args.project_cost
	
	W, I, communities, gamma_pos, success = simulation(NUM_AGENTS=NUM_AGENTS, STEPS=STEPS, PROJECT_COST=PROJECT_COST)