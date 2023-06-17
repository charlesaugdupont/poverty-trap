import numpy as np
import networkx as nx

from pymarkowitz import Optimizer

import random
import pickle

from tqdm import tqdm
from multiprocessing import Pool


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


def iso(x, eta=0.5):
    """
	Isoelastic utility function.
	"""
    return x**(1-eta) / (1-eta)


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
	return - (U(w*x, A=A, gamma=gamma) + V(sum(w*(1-x)*investment_returns)))

#################################################################################################

# Simulation

def simulation(NUM_AGENTS=1000, STEPS=50, SAFE_RETURN=1.10, DEFAULT_GAMMA=2.1,
			   PROJECT_COST=3.0, RL=1.2, RR=1.5,  W0=0.8, W1=1.2, NUM_GAMBLE_SAMPLES=1000, seed=None,
			   graph=None, graph_type="powerlaw_cluster", graph_args={"m":2, "p":0.5}):
	"""
	Runs ABM model.
	Args:
		NUM_AGENTS    	   : number of agents
		STEPS         	   : number of steps
		SAFE_RETURN   	   : safe return coefficient (> 1.0)
		DEFAULT_A     	   : parameter used in utility functions
		DEFAULT_GAMMA 	   : parameter used in utility functions
		PROJECT_COST  	   : minimum cost for project to be undertaken
		RL
		RR
		W0            	   : left bound for uniform random wealth initialization
		W1            	   : right bound for uniform random wealth initialization
		NUM_GAMBLE_SAMPLES : number of random samples for cumulative prospect theory utility
		seed
		graph 		  	   : NetworkX graph
		graph_type    	   : type of graph to use
		graph_args    	   : arguments for graph construction, specific to graph type passed
	Returns:
		WEALTH      : (STEPS, NUM_AGENTS) array containing wealth levels of agents at each iteration
		communities : dict from community ID to list of members
	"""
	with open("rfr.pickle", "rb") as f:
		response_surface_model = pickle.load(f)

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

	# global attributes
	GAMBLES = generate_gambles(len(communities), left=RL, right=RR)
	GAMBLES.append({"outcomes":[SAFE_RETURN, 0.0], "probs":[1.0, 0.0]})
	GAMBLE_PRIOR_SAMPLES = np.zeros((NUM_GAMBLE_SAMPLES, len(GAMBLES)))
	for i,g in enumerate(GAMBLES):
		GAMBLE_PRIOR_SAMPLES[:,i] = np.random.choice(g["outcomes"], NUM_GAMBLE_SAMPLES, p=g["probs"])
	GAMBLE_OBSERVED_SAMPLES = np.zeros((STEPS, len(GAMBLES)))
	GAMBLES_PRIOR_MU  = np.mean(GAMBLE_PRIOR_SAMPLES, axis=0)
	GAMBLES_PRIOR_COV = np.cov(GAMBLE_PRIOR_SAMPLES, rowvar=False)
	GAMBLE_RANDOM_RETURNS = np.row_stack([[get_gamble_returns(P, size=STEPS) for P in GAMBLES]])
	GAMBLE_SUCCESS = np.zeros((len(GAMBLES)))

	# agent attributes
	CONSUMPTION = np.zeros((STEPS, NUM_AGENTS))
	INCOME = np.zeros((STEPS, NUM_AGENTS))
	WEALTH = np.random.uniform(W0, W1, size=(STEPS+1, NUM_AGENTS))
	ATTENTION = np.random.uniform(size=NUM_AGENTS)
	RISK_AVERSION = np.random.uniform(0, 100, size=(NUM_AGENTS))


	A_VALUES = np.random.uniform(0.02, 0.04, size=NUM_AGENTS)
	# ETA = np.random.uniform(1, 30, size=NUM_AGENTS)
	# ETA_SCALE = np.random.randint(1.1, 2.0, size=NUM_AGENTS)


	# optimal portfolios, and expected returns for agents
	PORTFOLIO = compute_optimal_portfolios(NUM_AGENTS, len(communities)+1, GAMBLES_PRIOR_MU, GAMBLES_PRIOR_COV, RISK_AVERSION, community_membership)
	AGENT_EXPECTED_RETURNS = [PORTFOLIO[i][community_membership[i]] * GAMBLES_PRIOR_MU[community_membership[i]] for i in range(NUM_AGENTS)]

	# simulation
	print("Performing time stepping...")
	for step in tqdm(range(STEPS)):

		# compute updated portfolios and expected returns every 10 steps with attention weighting
		if step > 0 and (step) % 10 == 0:
			print(f"Updating portfolios... (step = {step})")
			recent_samples = GAMBLE_OBSERVED_SAMPLES[:step,:]
			MU  = np.mean(recent_samples, axis=0)
			COV = np.cov(recent_samples, rowvar=False)
			NEW_PORTFOLIO = compute_optimal_portfolios(NUM_AGENTS, len(communities)+1, MU, COV, RISK_AVERSION, community_membership)
			WEIGHTED_PORTFOLIO = PORTFOLIO = np.multiply((1-ATTENTION)[:,np.newaxis], PORTFOLIO) + np.multiply(ATTENTION[:,np.newaxis], NEW_PORTFOLIO)
			AGENT_EXPECTED_RETURNS = [(1-ATTENTION[i])*GAMBLES_PRIOR_MU[community_membership[i]]*PORTFOLIO[i][community_membership[i]] + \
			     					  ATTENTION[i]*MU[community_membership[i]]*WEIGHTED_PORTFOLIO[i][community_membership[i]] 
			     					  for i in range(NUM_AGENTS)]
			# update portfolios to the weighted ones
			PORTFOLIO = WEIGHTED_PORTFOLIO
			

		# agents choose consumption, and we compute contributions to each project
		stack = np.row_stack([[WEALTH[step][i], sum(AGENT_EXPECTED_RETURNS[i]), A_VALUES[i]] for i in range(NUM_AGENTS)]).reshape(NUM_AGENTS,3)

		#CONSUMPTION[step] = response_surface_model.predict(stack)
		CONSUMPTION[step] = np.random.uniform(0.1, 0.2, size=NUM_AGENTS)

		invested_wealth = WEALTH[step] * (1-CONSUMPTION[step])
		project_contributions = invested_wealth @ PORTFOLIO

		# get gamble returns
		successful_gambles = project_contributions >= PROJECT_COST
		GAMBLE_SUCCESS += successful_gambles
		returns = successful_gambles * GAMBLE_RANDOM_RETURNS[:,step]
		GAMBLE_OBSERVED_SAMPLES[step] = returns

		# update agent wealth and income
		INCOME[step] = np.multiply(invested_wealth[:,np.newaxis], PORTFOLIO) @ returns
		WEALTH[step+1] = INCOME[step]

	return WEALTH, INCOME, CONSUMPTION, ATTENTION, RISK_AVERSION, PORTFOLIO, A_VALUES, GAMBLE_SUCCESS, communities


def multithread_portfolio(arg_tuple):
	mu, cov, risk_aversion = arg_tuple
	optimizer = Optimizer(mu, cov)
	optimizer.add_objective("efficient_frontier", aversion=risk_aversion)
	optimizer.add_constraint("weight", weight_bound=(0,1), leverage=1)
	optimizer.solve()
	return optimizer.weight_sols


def compute_optimal_portfolios(NUM_AGENTS, num_projects, mu, cov, risk_aversion, community_membership):
	args = [(mu[community_membership[i]], cov[community_membership[i],:][:,community_membership[i]], risk_aversion[i]) for i in range(NUM_AGENTS)]
	with Pool() as pool:
		results = pool.map(multithread_portfolio, args)
	results = np.array(results, dtype=object)
	PORTFOLIO = np.zeros((NUM_AGENTS, num_projects))
	for i,portfolio in enumerate(results):
		PORTFOLIO[i][community_membership[i]] = portfolio
	return PORTFOLIO

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