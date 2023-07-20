import random
import numpy as np
import networkx as nx
from pymarkowitz import Optimizer


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
			   gain_right=1.7, 
			   prob_left=0.3,
			   init_wealth_scale=0.02,
			   risk_scale=4.00,
			   poisson_scale=12,
			   NUM_GAMBLE_SAMPLES=1000, 
			   seed=None,
			   communities=None,
			   community_membership=None,
			   graph=None,
			   graph_type="powerlaw_cluster", 
			   graph_args={"m":2, "p":0.5}):
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
		risk_scale           : uniform spread for risk aversion distribution
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

	# construct graph
	G = graph or build_graph(NUM_AGENTS, graph_type, graph_args)

	# extract communities and community membership
	communities = communities or get_communities(G)
	community_membership = community_membership or get_community_membership(G, communities)

	# generate a random gamble for each communitiy and append safe asset "gamble"
	GAMBLES = generate_gambles(len(communities), gain_right_bound=gain_right, prob_left=prob_left)
	GAMBLES.append({"outcomes":[SAFE_RETURN, 0.0], "probs":[1.0, 0.0]})

	# generate some prior samples, compute mean and covariance
	GAMBLE_PRIOR_SAMPLES = np.zeros((NUM_GAMBLE_SAMPLES, len(GAMBLES)))
	for i,g in enumerate(GAMBLES):
		GAMBLE_PRIOR_SAMPLES[:,i] = np.random.choice(g["outcomes"], NUM_GAMBLE_SAMPLES, p=g["probs"])
	GAMBLES_PRIOR_MU  = np.mean(GAMBLE_PRIOR_SAMPLES, axis=0)
	GAMBLES_PRIOR_COV = np.cov(GAMBLE_PRIOR_SAMPLES, rowvar=False)	
	assert SAFE_RETURN <= np.min(GAMBLES_PRIOR_MU)

	# generate some random gamble returns
	GAMBLE_RANDOM_RETURNS = np.row_stack([[get_gamble_returns(P, size=STEPS) for P in GAMBLES]])

	# array to keep track of actual empirical gamble returns
	GAMBLE_OBSERVED_SAMPLES = np.zeros((STEPS, len(GAMBLES)), dtype=np.float16)

	# agent attributes
	CONSUMPTION = np.zeros((STEPS, NUM_AGENTS), dtype=np.float16)
	WEALTH = np.zeros((STEPS+1, NUM_AGENTS), dtype=np.float16)
	WEALTH[0,:] = np.random.normal(1, init_wealth_scale, size=NUM_AGENTS)
	ATTENTION = np.random.uniform(size=NUM_AGENTS).astype(np.float16)
	RISK_AVERSION = np.random.uniform(20 - risk_scale, 20 + risk_scale, size=(NUM_AGENTS)).astype(np.float16)

	# generate some Poisson distributed portfolio update times
	POISSON_TIMES = np.random.poisson(poisson_scale, size=(NUM_AGENTS, 12))
	UPDATE_TIMES = {k:list(v) for k,v in enumerate(np.cumsum(POISSON_TIMES, axis=1))}

	# initialize portfolios and compute expected returns for each agent
	PORTFOLIOS = initialize_portfolios(NUM_AGENTS, len(communities)+1, GAMBLES_PRIOR_MU, GAMBLES_PRIOR_COV, RISK_AVERSION, community_membership)
	AGENT_EXPECTED_RETURNS = [PORTFOLIOS[i][community_membership[i]] * GAMBLES_PRIOR_MU[community_membership[i]] for i in range(NUM_AGENTS)]
	ALL_PORTFOLIOS = {i:[PORTFOLIOS[i][community_membership[i]]] for i in range(NUM_AGENTS)}

	# RUN SIMULATION
	for step in range(STEPS):

		# check for portfolio updates after a "burn-in" period of 5 steps
		if step >= 5:
			recent_samples = GAMBLE_OBSERVED_SAMPLES[:step,:]
			MU  = np.mean(recent_samples, axis=0)
			COV = np.cov(recent_samples, rowvar=False)
			
			for i in range(NUM_AGENTS):
				# check if agent needs to be updated at current step
				if step in UPDATE_TIMES[i]:
					comm_mem = community_membership[i]
					portfolio_update(i, comm_mem, RISK_AVERSION[i], ATTENTION[i], MU[comm_mem],
		      						 COV[comm_mem,:][:,comm_mem], GAMBLES_PRIOR_MU[comm_mem],
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

	return WEALTH, ATTENTION, RISK_AVERSION, ALL_PORTFOLIOS, UPDATE_TIMES, communities, GAMBLE_OBSERVED_SAMPLES

def portfolio_update(i, community_membership, risk_aversion, attention, mu, cov, 
		     		 gambles_prior_mu, AGENT_EXPECTED_RETURNS, PORTFOLIOS, ALL_PORTFOLIOS):
	"""
	Update an agent's portfolio and expected portfolio return.
	"""
	# solve portfolio optimization
	optimizer = Optimizer(mu, cov)
	optimizer.add_objective("efficient_frontier", aversion=risk_aversion)
	optimizer.add_constraint("weight", weight_bound=(0,1), leverage=1)
	optimizer.solve()

	# update portfolio
	updated_portfolio = (1-attention)*ALL_PORTFOLIOS[i][0] + attention*optimizer.weight_sols
	ALL_PORTFOLIOS[i].append(updated_portfolio)
	NEW_PORTFOLIO = np.zeros(PORTFOLIOS.shape[1])
	NEW_PORTFOLIO[community_membership] = updated_portfolio
	PORTFOLIOS[i] = NEW_PORTFOLIO

	# update expected portfolio return
	updated_mu = (1-attention)*gambles_prior_mu + attention*mu
	AGENT_EXPECTED_RETURNS[i] = updated_mu * PORTFOLIOS[i][community_membership]	

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