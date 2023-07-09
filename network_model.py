import numpy as np
import networkx as nx

from pymarkowitz import Optimizer

import random

#from tqdm import tqdm


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
	membership = {k:np.array(list(v)+[len(communities)]) for k,v in membership.items()}
	return membership

#################################################################################################

# Gambles

def generate_gambles(N, gain_right_bound, prob_left=0.3):
	"""
	Generate N gambles with 2 outcomes.
	"""
	probs     = np.random.uniform(prob_left, 1-prob_left, N)
	outcomes1 = np.random.uniform(0.90, 0.95, N)
	outcomes2 = np.random.uniform(1.6, gain_right_bound, N)

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

# Simulation

def simulation(NUM_AGENTS=1250, 
	       	   STEPS=50,
			   SAFE_RETURN=1.10,
			   PROJECT_COST=3.0,  
			   gain_right=1.7, 
			   ALPHA_BETA=0.8,
			   prob_left=0.3,
			   init_wealth_scale=0.02,
			   risk_scale=4.00,
			   poisson_scale=12,
			   NUM_GAMBLE_SAMPLES=1000, 
			   seed=None,
			   graph=None,
			   graph_type="powerlaw_cluster", 
			   graph_args={"m":2, "p":0.5}):
	"""
	Runs ABM model.
	Args:
		NUM_AGENTS    	   : number of agents
		STEPS         	   : number of steps
		SAFE_RETURN   	   : safe return coefficient (> 1.0)
		PROJECT_COST  	   : minimum cost for project to be undertaken
		gain_right		   : right bound for generating gamble gains
		prob_left 		   : left uniform bound for generating gamble branch probabilities
		ALPHA			   : production function parameter used to compute optimal consumption
		BETA               : time discounting factor used to compute optimal consumption
		NUM_GAMBLE_SAMPLES : number of random samples for cumulative prospect theory utility
		seed			   : random seed
		graph 		  	   : NetworkX graph
		graph_type    	   : type of graph
		graph_args    	   : arguments for graph construction, specific to graph_type
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
	#print(f"{len(communities)} communities.")

	# get community membership of each agent
	community_membership = get_community_membership(G, communities)

	# generate random gambles and append safe asset
	GAMBLES = generate_gambles(len(communities), gain_right_bound=gain_right, prob_left=prob_left)
	GAMBLES.append({"outcomes":[SAFE_RETURN, 0.0], "probs":[1.0, 0.0]})

	# generate some prior samples, and compute mean and covariance
	GAMBLE_PRIOR_SAMPLES = np.zeros((NUM_GAMBLE_SAMPLES, len(GAMBLES)))
	for i,g in enumerate(GAMBLES):
		GAMBLE_PRIOR_SAMPLES[:,i] = np.random.choice(g["outcomes"], NUM_GAMBLE_SAMPLES, p=g["probs"])
	GAMBLES_PRIOR_MU  = np.mean(GAMBLE_PRIOR_SAMPLES, axis=0)
	GAMBLES_PRIOR_COV = np.cov(GAMBLE_PRIOR_SAMPLES, rowvar=False)	
	assert SAFE_RETURN <= np.min(GAMBLES_PRIOR_MU[:-1])

	# generate some random gamble returns
	GAMBLE_RANDOM_RETURNS = np.row_stack([[get_gamble_returns(P, size=STEPS) for P in GAMBLES]])

	# array to keep track of actual empirical gamble returns
	GAMBLE_OBSERVED_SAMPLES = np.zeros((STEPS, len(GAMBLES)))

	# agent attributes
	CONSUMPTION = np.zeros((STEPS, NUM_AGENTS))
	WEALTH = np.zeros((STEPS+1, NUM_AGENTS))
	WEALTH[0,:] = np.random.normal(1, init_wealth_scale, size=NUM_AGENTS)
	ATTENTION = np.random.uniform(size=NUM_AGENTS).astype(np.float32)
	RISK_AVERSION = np.random.uniform(20 - risk_scale, 20 + risk_scale, size=(NUM_AGENTS)).astype(np.float32)

	# generate some Poisson distributed portfolio update times (first time is at least 3)
	MIN_UPDATE_TIME = 5
	POISSON_TIMES = np.random.poisson(poisson_scale, size=(NUM_AGENTS, 12))
	POISSON_TIMES[:,0] = np.maximum(MIN_UPDATE_TIME, POISSON_TIMES[:,0])
	UPDATE_TIMES = {k:list(v) for k,v in enumerate(np.cumsum(POISSON_TIMES, axis=1))}

	# initialize portfolios and compute expected returns for each agent
	PORTFOLIOS = initialize_portfolios(NUM_AGENTS, len(communities)+1, GAMBLES_PRIOR_MU, GAMBLES_PRIOR_COV, RISK_AVERSION, community_membership)
	AGENT_EXPECTED_RETURNS = [PORTFOLIOS[i][community_membership[i]] * GAMBLES_PRIOR_MU[community_membership[i]] for i in range(NUM_AGENTS)]
	ALL_PORTFOLIOS = {i:[PORTFOLIOS[i][community_membership[i]]] for i in range(NUM_AGENTS)}

	# RUN SIMULATION
	#print("Performing time stepping...")
	for step in range(STEPS):

		# check for portfolio updates
		if step >= MIN_UPDATE_TIME:
			recent_samples = GAMBLE_OBSERVED_SAMPLES[:step,:]
			MU  = np.mean(recent_samples, axis=0)
			COV = np.cov(recent_samples, rowvar=False)
			for i in range(NUM_AGENTS):
				if step in UPDATE_TIMES[i]:
					comm_mem = community_membership[i]
					portfolio_update(i, comm_mem, RISK_AVERSION[i], ATTENTION[i], MU[comm_mem],
		      						 COV[comm_mem,:][:,comm_mem], GAMBLES_PRIOR_MU[comm_mem],
								     AGENT_EXPECTED_RETURNS, PORTFOLIOS)
					ALL_PORTFOLIOS[i].append(PORTFOLIOS[i][comm_mem])

		# agents choose consumption, and we compute contributions to each project
		expected_returns = np.array([sum(AGENT_EXPECTED_RETURNS[i]) for i in range(NUM_AGENTS)])
		CONSUMPTION[step] = (1-ALPHA_BETA)*WEALTH[step]*expected_returns
		invested_wealth = WEALTH[step] - CONSUMPTION[step]
		project_contributions = invested_wealth @ PORTFOLIOS

		# get gamble returns
		successful_gambles = project_contributions >= PROJECT_COST

		# safe asset has guaranteed return
		successful_gambles[-1] = True
		returns = successful_gambles * GAMBLE_RANDOM_RETURNS[:,step]
		GAMBLE_OBSERVED_SAMPLES[step] = returns

		# update agent wealth
		WEALTH[step+1] = np.minimum(1e9, np.multiply(invested_wealth[:,np.newaxis], PORTFOLIOS) @ returns)

	return WEALTH, CONSUMPTION, ATTENTION, RISK_AVERSION, ALL_PORTFOLIOS, UPDATE_TIMES, communities, GAMBLE_OBSERVED_SAMPLES


def portfolio_update(i, community_membership, risk_aversion, attention, mu, cov, gambles_prior_mu, AGENT_EXPECTED_RETURNS, PORTFOLIOS):
	NEW_PORTFOLIO = np.zeros(PORTFOLIOS.shape[1])
	optimizer = Optimizer(mu, cov)
	optimizer.add_objective("efficient_frontier", aversion=risk_aversion)
	optimizer.add_constraint("weight", weight_bound=(0,1), leverage=1)
	optimizer.solve()
	NEW_PORTFOLIO[community_membership] = optimizer.weight_sols
	AGENT_EXPECTED_RETURNS[i] = (1-attention)*gambles_prior_mu*PORTFOLIOS[i][community_membership] + \
								attention*mu*NEW_PORTFOLIO[community_membership]
	PORTFOLIOS[i] = (1-attention)*PORTFOLIOS[i] + attention*NEW_PORTFOLIO


def initialize_portfolios(NUM_AGENTS, num_projects, mu, cov, risk_aversion, community_membership):
	INITIAL_PORTFOLIOS = np.zeros((NUM_AGENTS, num_projects))
	for i in range(NUM_AGENTS):
		M, C, R = mu[community_membership[i]], cov[community_membership[i],:][:,community_membership[i]], risk_aversion[i]
		optimizer = Optimizer(M, C)
		optimizer.add_objective("efficient_frontier", aversion=R)
		optimizer.add_constraint("weight", weight_bound=(0,1), leverage=1)
		optimizer.solve()
		INITIAL_PORTFOLIOS[i][community_membership[i]] = optimizer.weight_sols
	return INITIAL_PORTFOLIOS

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