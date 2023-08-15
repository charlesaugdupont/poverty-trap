import random
import numpy as np
import networkx as nx
from cpt_utility import *
from cpt_optimizer import *

#################################################################################################

# Network Construction

def build_graph(n, graph_type, graph_args):
	"""
	Constructs networkx graph.
	"""
	if graph_type == "powerlaw_cluster":
		G = nx.powerlaw_cluster_graph(n=n, **graph_args) 
	elif graph_type == "random":
		G = nx.random_regular_graph(n=n, **graph_args)
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

#################################################################################################

# Simulation

def simulation(NUM_AGENTS=1250, 
	       	   STEPS=50,
			   SAFE_RETURN=1.10,
			   PROJECT_COST=3.0,  
			   ALPHA_BETA=0.8,
			   gain_right=2.0, 
			   prob_left=0.3,
			   poisson_scale=14.0,
			   NUM_GAMBLE_SAMPLES=1000, 
			   seed=None,
			   communities=None,
			   community_membership=None,
			   INIT_WEALTH_VALUES=None):
	"""
	Runs ABM model.
	Args:
		NUM_AGENTS    	     : number of agents
		STEPS         	     : number of steps
		SAFE_RETURN   	     : safe return coefficient (> 1.0)
		PROJECT_COST  	     : minimum cost for project to be undertaken
		gain_right		     : right bound for generating gamble gains
		ALPHA_BETA		     : constant used to compute optimal consumption
		prob_left 		     : left uniform bound for generating gamble branch probabilities
		init_wealth_scale    : standard deviation for initial wealth distribution
		poisson_scale        : mean time between portfolio updates
		NUM_GAMBLE_SAMPLES   : number of random samples for cumulative prospect theory utility
		seed			     : random seed
		communities  	   	 : graph communities
		community_membership : mapping from node to communities it is a part of
		graph 		  	     : NetworkX graph
		graph_type    	     : type of graph
		graph_args    	     : arguments for graph construction, specific to graph_type
	"""
	# RNG
	if seed:
		random.seed(seed)
		np.random.seed(seed)

	# generate a random gamble for each communitiy and append safe asset "gamble"
	GAMBLES = generate_gambles(len(communities), gain_right_bound=gain_right, prob_left=prob_left)
	GAMBLES.append({"outcomes":[SAFE_RETURN, 0.0], "probs":[1.0, 0.0]})

	# generate some prior samples, compute mean and covariance
	GAMBLE_PRIOR_SAMPLES = np.zeros((NUM_GAMBLE_SAMPLES, len(GAMBLES)))
	for i,g in enumerate(GAMBLES):
		GAMBLE_PRIOR_SAMPLES[:,i] = np.random.choice(g["outcomes"], NUM_GAMBLE_SAMPLES, p=g["probs"])
	GAMBLES_PRIOR_MU  = np.mean(GAMBLE_PRIOR_SAMPLES, axis=0)
	assert SAFE_RETURN <= np.min(GAMBLES_PRIOR_MU)

	# generate some random gamble returns
	GAMBLE_RANDOM_RETURNS = np.row_stack([[get_gamble_returns(P, size=STEPS) for P in GAMBLES]])

	# array to keep track of actual empirical gamble returns
	GAMBLE_OBSERVED_SAMPLES = np.zeros((STEPS, len(GAMBLES)), dtype=np.float16)

	# agent attributes
	CONSUMPTION = np.zeros((STEPS, NUM_AGENTS), dtype=np.float16)
	WEALTH = np.zeros((STEPS+1, NUM_AGENTS), dtype=np.float16)
	WEALTH[0,:] = INIT_WEALTH_VALUES
	ATTENTION = np.random.uniform(size=NUM_AGENTS).astype(np.float16)

	# CPT utilities
	GAMMA_POS = np.random.uniform(5, 30, size=NUM_AGENTS).round(2)
	GAMMA_NEG = np.random.uniform(31, 70, size=NUM_AGENTS).round(2)
	DELTA_POS = np.random.uniform(0.50, 0.70, size=NUM_AGENTS).round(2)
	DELTA_NEG = np.random.uniform(0.71, 0.90, size=NUM_AGENTS).round(2)

	UTILITIES = [CPTUtility(gamma_pos=GAMMA_POS[i], 
			 				gamma_neg=GAMMA_NEG[i], 
							delta_pos=DELTA_POS[i], 
							delta_neg=DELTA_NEG[i]) for i in range(NUM_AGENTS)]

	# generate some Poisson distributed portfolio update times
	POISSON_TIMES = np.random.poisson(poisson_scale, size=(NUM_AGENTS, 12))
	UPDATE_TIMES = {k:list(v) for k,v in enumerate(np.cumsum(POISSON_TIMES, axis=1))}

	# initialize portfolios and compute expected returns for each agent
	PORTFOLIOS = initialize_portfolios(NUM_AGENTS, len(communities)+1, UTILITIES, GAMBLE_PRIOR_SAMPLES, community_membership)
	AGENT_EXPECTED_RETURNS = [PORTFOLIOS[i][community_membership[i]] * GAMBLES_PRIOR_MU[community_membership[i]] for i in range(NUM_AGENTS)]
	ALL_PORTFOLIOS = {i:[PORTFOLIOS[i][community_membership[i]]] for i in range(NUM_AGENTS)}

	# RUN SIMULATION
	for step in range(STEPS):

		# check for portfolio updates after a "burn-in" period of 5 steps
		if step >= 5:
			recent_samples = GAMBLE_OBSERVED_SAMPLES[:step,:]
			MU  = np.mean(recent_samples, axis=0)			
			for i in range(NUM_AGENTS):
				# check if agent needs to be updated at current step
				if step in UPDATE_TIMES[i]:
					comm_mem = community_membership[i]
					portfolio_update(i, UTILITIES[i], recent_samples[:,comm_mem], 
						      		 comm_mem, ATTENTION[i], MU[comm_mem], GAMBLES_PRIOR_MU[comm_mem],
									 AGENT_EXPECTED_RETURNS, PORTFOLIOS, ALL_PORTFOLIOS)

		# agents choose consumption, and we compute contributions to each project
		expected_returns = np.array([sum(AGENT_EXPECTED_RETURNS[i]) for i in range(NUM_AGENTS)])
		CONSUMPTION[step] = (1-ALPHA_BETA)*WEALTH[step]*expected_returns
		invested_wealth = WEALTH[step] - CONSUMPTION[step]
		project_contributions = invested_wealth @ PORTFOLIOS

		# get gamble returns
		successful_gambles = project_contributions >= PROJECT_COST
		successful_gambles[-1] = True # safe asset has guaranteed return
		returns = (successful_gambles * GAMBLE_RANDOM_RETURNS[:,step]).astype(np.float16)
		GAMBLE_OBSERVED_SAMPLES[step] = returns

		# update agent wealth
		WEALTH[step+1] = np.minimum(6e4, np.multiply(invested_wealth[:,np.newaxis], PORTFOLIOS) @ returns)

	return WEALTH, ATTENTION, UTILITIES, ALL_PORTFOLIOS, UPDATE_TIMES, communities, GAMBLE_OBSERVED_SAMPLES

def portfolio_update(i, utility, gamble_returns, community_membership, attention, mu, 
		     		 gambles_prior_mu, AGENT_EXPECTED_RETURNS, PORTFOLIOS, ALL_PORTFOLIOS):
	"""
	Update an agent's portfolio and expected portfolio return.
	Args:
		i					   : agent index
		utility				   : cumulative prospect theory utility of the agent
		gamble_returns		   : array of observed project returns relevant to the agent; shape is (num steps so far, # projects)
		community_membership   : array of indices of the communities that the agent is a part of
		attention			   : agent attention parameter
		mu 					   : mean vector of observed project returns
		gambles_prior_mu	   : mean vector of prior project samples
		AGENT_EXPECTED_RETURNS : vector of expected portfolio return for all agents
		PORTFOLIOS			   : dictionary from agent index to current agent portfolio
		ALL_PORTFOLIOS		   : dictionary from agent index to historical list of agent's portfolios
	"""
	# initialize new empty portfolio 
	NEW_PORTFOLIO = np.zeros(PORTFOLIOS.shape[1])

	# instantiate optimizer with the CPT utility
	mv = MeanVarianceFrontierOptimizer(utility)
	reps = 0
	while mv._weights is None:
		reps += 1
		if reps == 30:
			print("Reached max optimization repeat attempts!")
			mv._weights = ALL_PORTFOLIOS[i][-1]
			break
		else:
			# if optimization fails, retry up to 30 times
			try:
				mv.optimize(gamble_returns-1)
			except:
				continue
	
	# construct updated portfolio using attention mechanism
	updated_portfolio = (1-attention)*ALL_PORTFOLIOS[i][0] + attention*mv.weights
	ALL_PORTFOLIOS[i].append(updated_portfolio)
	NEW_PORTFOLIO = np.zeros(PORTFOLIOS.shape[1])
	NEW_PORTFOLIO[community_membership] = updated_portfolio
	PORTFOLIOS[i] = NEW_PORTFOLIO

	# update expected portfolio return
	updated_mu = (1-attention)*gambles_prior_mu + attention*mu
	AGENT_EXPECTED_RETURNS[i] = updated_mu * PORTFOLIOS[i][community_membership]

def initialize_portfolios(NUM_AGENTS, num_projects, UTILITIES, GAMBLE_SAMPLES, community_membership):
	"""
	Initializes all agents' portfolios.
	Args:
		NUM_AGENTS 	 		 : number of agents
		num_projects 		 : number of projects
		UTILITIES			 : list of agent CPT utility functions
		GAMBLE_SAMPLES 		 : array of prior project samples; shape is (1000, num_projects)
		community_membership : dictionary from agent index to array of indices of the communities that the agent is a part of
	"""
	INITIAL_PORTFOLIOS = np.zeros((NUM_AGENTS, num_projects))
	for i in range(NUM_AGENTS):
		mv = MeanVarianceFrontierOptimizer(UTILITIES[i])
		mv.optimize(GAMBLE_SAMPLES[:,community_membership[i]]-1)
		INITIAL_PORTFOLIOS[i][community_membership[i]] = mv.weights
	return INITIAL_PORTFOLIOS