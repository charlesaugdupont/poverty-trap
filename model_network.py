import numpy as np
import networkx as nx
from tqdm import tqdm
from model_banerjee import U,V
from scipy.stats import gengamma
from scipy.optimize import minimize

from multiprocessing import Pool


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


def get_community_membership(adjacency, node_community_map):
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
	probs     = np.random.uniform(0.2, 0.8, N)
	outcomes1 = np.random.uniform(0.9, 0.95, N)
	outcomes2 = np.random.uniform(2.0, 2.5, N)

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

def utility(x, w, project_allocations, project_returns, risk, safe_return, A, gamma):
	"""
	Compute expected utility.
	Args:
		x                   : consumption proportion
		w                   : wealth level of agent
		project_allocations : allocation of savings to risky projects
		project_returns     : expected project return
		risk                : parameter controlling level of risk-aversion
		safe_return         : safe return amount
		A                   : utility function parameter
		gamma               : utility function parameter
	Returns:
		utility
	"""
	# utility from consumption
	consumption_utility = U(w*x, A=A, gamma=gamma)
	
	# expected utility from projects
	i = 1-x
	project_utility = V(w*i*(1-risk)*safe_return + sum(w*i*risk*project_allocations*project_returns))
	
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

def simulation(NUM_AGENTS=500, STEPS=50, SAFE_RETURN=1.1, DEFAULT_A=1.2, DEFAULT_GAMMA=2.1,
			   PROJECT_COST=3.0, W0=0.8, W1=1.2, R0=0.3, R1=0.7, graph=None,
			   graph_type="powerlaw_cluster", graph_args={"m":2, "p":0.5}):
	"""
	Runs ABM model.
	Args:
		NUM_AGENTS    : number of agents
		STEPS         : number of steps
		SAFE_RETURN   : safe return coefficient (> 1.0)
		DEFAULT_A     : parameter used in utility functions
		DEFAULT_GAMMA : parameter used in utility functions
		PROJECT_COST  : minimum cost for project to be undertaken
		W0            : left bound for uniform random wealth initialization
		W1            : right bound for uniform random wealth initialization
		R0            : left bound for uniform random risk initialization
		R1            : right bound for uniform random risk initialization
		graph 		  : NetworkX graph
		graph_type    : type of graph to use
		graph_args    : arguments for graph construction, specific to graph type passed
	Returns:
		WEALTH      : (STEPS, NUM_AGENTS) array containing wealth levels of agents at each iteration
		communities : dict from community ID to list of members
	"""
	multiprocess = NUM_AGENTS >= 10000

	# construct graph and adjacency matrix
	G = graph or build_graph(NUM_AGENTS, graph_type, graph_args)
	adjacency = get_adjacency(G)
	
	# extract communities
	communities = get_communities(G)
	print(f"{len(communities)} communities.")
	
	# get community membership of nodes
	node_community_map = get_node_community_map(communities)
	community_membership = get_community_membership(adjacency, node_community_map)
	communities = {c:[] for c in range(len(communities))}
	for i, comms in community_membership.items():
		for c in comms:
			communities[c].append(i)

	# global attributes
	GAMBLES = generate_gambles(len(communities))
	GAMBLE_RETURNS = np.row_stack([[get_gamble_returns(P, size=STEPS) for P in GAMBLES]])
	gamble_success   = np.zeros((len(GAMBLES)))
	gamble_averages  = np.zeros((len(GAMBLES)))
	gamble_variances = np.zeros((len(GAMBLES)))
	# compute expected value and variance of each gamble
	for i,g in enumerate(GAMBLES):
		gamble_averages[i]  = np.average(g["outcomes"], weights=g["probs"])
		gamble_variances[i] = np.average((g["outcomes"]-gamble_averages[i])**2, weights=g["probs"])
			
	# agent attributes
	INCOME = np.zeros((STEPS, NUM_AGENTS))
	WEALTH = np.random.uniform(W0, W1, (STEPS+1, NUM_AGENTS))
	RISK   = np.random.uniform(R0, R1, NUM_AGENTS)
	C      = np.zeros((NUM_AGENTS))
	ALLOC  = []
	
	# compute optimal portfolios for agents
	for i in range(NUM_AGENTS):
		mu      = gamble_averages[community_membership[i]]
		cov     = np.diag(gamble_variances[community_membership[i]])
		cov_inv = np.linalg.inv(cov)
		ones    = np.ones((len(mu),))
		val     = cov_inv @ ((mu - SAFE_RETURN) * ones)
		val     = val / (ones @ val)
		ALLOC.append(val)

	# simulation
	for step in tqdm(range(STEPS)):

		project_contributions = np.zeros((len(GAMBLES)))

		# all agents perform optimization step and we sum up project contributions

		if multiprocess:
			args = [(WEALTH[step][i], ALLOC[i], 
					gamble_averages[community_membership[i]],
					RISK[i], DEFAULT_A, DEFAULT_GAMMA, SAFE_RETURN)
					for i in range(NUM_AGENTS)]
			with Pool() as pool:
				results = pool.map(optimize_utility, args)
			C = np.array(results)
			for i in range(NUM_AGENTS):
				project_contributions[community_membership[i]] += \
									np.sum(WEALTH[step][i]*ALLOC[i]*RISK[i]*(1-C[i]))  
		else:
			# serial approach (better than multiprocessing for O(10^3) agents, but worse for >= O(10^4))
			for i in range(NUM_AGENTS):
				C[i] = minimize(utility, x0=0.5, bounds=[(0.05, 0.95)], method='SLSQP',
								args=(WEALTH[step][i], 
									ALLOC[i], 
									gamble_averages[community_membership[i]],
									RISK[i], 
									DEFAULT_A, 
									DEFAULT_GAMMA, 
									SAFE_RETURN)).x[0]
				project_contributions[community_membership[i]] += \
									np.sum(WEALTH[step][i]*ALLOC[i]*RISK[i]*(1-C[i])) 
		  

		# run projects
		risky_returns = np.zeros((len(GAMBLES)))
		for idx in range(len(GAMBLES)):
			if project_contributions[idx] >= PROJECT_COST:
				risky_returns[idx] = GAMBLE_RETURNS[idx][step]
				gamble_success[idx] += 1
	
		for i in range(NUM_AGENTS):
			# new wealth = old wealth - consumption - risky investment + risky return + safe return
			I = 1-C[i]
			S = 1-RISK[i]
			agent_risky_return = np.array(risky_returns[community_membership[i]])
			
			# calculate income:
			# sum of risky investment returns - risky investment amount ("-1") + safe return
			INCOME[step][i] = sum(WEALTH[step][i]*I*RISK[i]*ALLOC[i]*(agent_risky_return-1)) + \
								  WEALTH[step][i]*I*S*SAFE_RETURN
			
			# new wealth = current wealth - consumption + income from safe/risky investments
			WEALTH[step+1][i]  = WEALTH[step][i] - WEALTH[step][i]*C[i] + INCOME[step][i]

	return WEALTH, INCOME, communities, RISK, gamble_success

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

	W, I, communities, risk, success = simulation(NUM_AGENTS=10000, STEPS=10, PROJECT_COST=0.5)