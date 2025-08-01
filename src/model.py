import random
from utils import *
from cpt import *
from tqdm.auto import tqdm


def simulation(
		NUM_AGENTS=1225, 
		STEPS=100,
		SAFE_RETURN=1.10,
		PROJECT_COSTS=None,  
		SAVING_PROP=0.8,
		GAIN_RIGHT=1.7,
		PROB_LEFT=0.3,
		POISSON_SCALE=10,
		NUM_GAMBLE_SAMPLES=2000,
		SEED=None,
		COMMUNITIES=None,
		COMMUNITY_MEMBERSHIP=None,
		INIT_WEALTH_VALUES=None):
	"""
	Runs ABM model.
	Args:
		NUM_AGENTS    	     : number of agents
		STEPS         	     : number of steps
		SAFE_RETURN   	     : safe return coefficient (> 1.0)
		PROJECT_COSTS  	     : array of project costs
		SAVING_PROP		     : saving propensity
		GAIN_RIGHT		     : right bound for generating gamble gains
		PROB_LEFT 		     : left uniform bound for generating gamble branch probabilities
		POISSON_SCALE        : mean time between portfolio updates
		NUM_GAMBLE_SAMPLES   : number of random samples for cumulative prospect theory utility
		SEED			     : random seed
		COMMUNITIES  	   	 : graph communities
		COMMUNITY_MEMBERSHIP : mapping from node to communities it is a part of
		INIT_WEALTH_VALUES   : array of initial agent wealth values
	"""
	# RNG
	if SEED:
		random.seed(SEED)
		np.random.seed(SEED)

	# generate a random gamble for each communitiy and append safe asset "gamble"
	gambles = generate_gambles(len(COMMUNITIES), gain_right_bound=GAIN_RIGHT, prob_left=PROB_LEFT)
	gambles.append({"outcomes":[SAFE_RETURN, 0.0], "probs":[1.0, 0.0]})

	# generate some prior samples, compute mean and covariance
	gamble_prior_samples = np.zeros((NUM_GAMBLE_SAMPLES, len(gambles)))
	for i,g in enumerate(gambles):
		gamble_prior_samples[:,i] = np.random.choice(g["outcomes"], NUM_GAMBLE_SAMPLES, p=g["probs"])
	assert SAFE_RETURN <= np.min(np.mean(gamble_prior_samples, axis=0).round(1))

	# generate some random gamble returns
	gamble_random_returns = np.row_stack([[get_gamble_returns(P, size=STEPS) for P in gambles]])

	# array to keep track of actual empirical gamble returns
	gamble_observed_samples = np.zeros((STEPS, len(gambles)), dtype=np.float16)

	# attributes and metrics
	consumption = np.zeros((STEPS, NUM_AGENTS), dtype=np.float16)
	investment = np.zeros((STEPS, NUM_AGENTS), dtype=np.float16)
	wealth = np.zeros((STEPS+1, NUM_AGENTS), dtype=np.float16)
	wealth[0,:] = INIT_WEALTH_VALUES
	attention = np.random.uniform(size=NUM_AGENTS).astype(np.float16)
	contributions = np.zeros((STEPS, len(COMMUNITIES)+1), dtype=np.float16)

	# CPT utilities
	gamma_pos = np.random.uniform(5, 30, size=NUM_AGENTS).round(2)
	gamma_neg = np.random.uniform(31, 70, size=NUM_AGENTS).round(2)
	delta_pos = np.random.uniform(0.50, 0.70, size=NUM_AGENTS).round(2)
	delta_neg = np.random.uniform(0.71, 0.90, size=NUM_AGENTS).round(2)
	utilities = [CPTUtility(gamma_pos=gamma_pos[i],
			 				gamma_neg=gamma_neg[i],
							delta_pos=delta_pos[i],
							delta_neg=delta_neg[i]) for i in range(NUM_AGENTS)]

	# generate some Poisson distributed portfolio update times
	poisson_times = np.random.poisson(POISSON_SCALE, size=(NUM_AGENTS, 12))
	update_times = {k:list(v) for k,v in enumerate(np.cumsum(poisson_times, axis=1))}

	# initialize portfolio for each agent
	portfolios = initialize_portfolios(NUM_AGENTS, len(COMMUNITIES)+1, utilities, gamble_prior_samples, COMMUNITY_MEMBERSHIP)
	all_portfolios = {i:[portfolios[i][COMMUNITY_MEMBERSHIP[i]]] for i in range(NUM_AGENTS)}

	# RUN SIMULATION
	for step in tqdm(range(STEPS)):

		# check for portfolio updates after a "burn-in" period of 5 steps
		if step >= 5:
			recent_samples = gamble_observed_samples[:step,:]
			for i in range(NUM_AGENTS):
				# check if agent needs to be updated at current step
				if step in update_times[i]:
					comm_mem = COMMUNITY_MEMBERSHIP[i]
					portfolio_update(i, utilities[i], recent_samples[:,comm_mem], 
						      		 comm_mem, attention[i], portfolios, all_portfolios)

		# agents choose consumption, and we compute contributions to each project
		consumption[step] = (1-SAVING_PROP)*wealth[step]
		invested_wealth = wealth[step] - consumption[step]
		investment[step] = invested_wealth
		project_contributions = invested_wealth @ portfolios
		contributions[step] = project_contributions

		# get gamble returns
		successful_gambles = project_contributions >= PROJECT_COSTS
		successful_gambles[-1] = True # safe asset has guaranteed return
		returns = (successful_gambles * gamble_random_returns[:,step]).astype(np.float16)
		gamble_observed_samples[step] = returns

		# update agent wealth
		wealth[step+1] = np.minimum(6e4, np.multiply(invested_wealth[:,np.newaxis], portfolios) @ returns)

	return wealth, investment, contributions, gamble_observed_samples, \
		   attention, utilities, all_portfolios, update_times


def portfolio_update(i, utility, gamble_returns, community_membership, 
					 attention, portfolios, all_portfolios):
	"""
	Update an agent's portfolio and expected portfolio return.
	Args:
		i					   : agent index
		utility				   : cumulative prospect theory utility of the agent
		gamble_returns		   : array of observed project returns relevant to the agent; shape is (num steps so far, # projects)
		community_membership   : array of indices of the communities that the agent is a part of
		attention			   : agent attention parameter
		PORTFOLIOS			   : dictionary from agent index to current agent portfolio
		all_portfolios		   : dictionary from agent index to historical list of agent's portfolios
	"""
	# instantiate optimizer with the CPT utility
	mv = MeanVarianceFrontierOptimizer(utility)
	reps = 0
	while mv._weights is None:
		reps += 1
		if reps == 30:
			print("Reached max optimization repeat attempts!")
			mv._weights = all_portfolios[i][-1]
			break
		else:
			# if optimization fails, retry up to 30 times
			try:
				mv.optimize(gamble_returns-1)
			except:
				continue
	
	# construct updated portfolio using attention mechanism
	updated_portfolio = (1-attention)*all_portfolios[i][0] + attention*mv.weights
	all_portfolios[i].append(updated_portfolio)

	# store the new portfolio
	new_portfolio = np.zeros(portfolios.shape[1])
	new_portfolio[community_membership] = updated_portfolio
	portfolios[i] = new_portfolio


def initialize_portfolios(NUM_AGENTS, num_projects, utilities, gamble_samples, COMMUNITY_MEMBERSHIP):
	"""
	Initializes all agents' portfolios.
	Args:
		NUM_AGENTS 	 		 : number of agents
		num_projects 		 : number of projects
		utilities			 : list of agent CPT utility functions
		gamble_samples 		 : array of prior project samples; shape is (1000, num_projects)
		COMMUNITY_MEMBERSHIP : dictionary from agent index to array of indices of the communities that the agent is a part of
	"""
	initial_portfolios = np.zeros((NUM_AGENTS, num_projects))
	for i in range(NUM_AGENTS):
		mv = MeanVarianceFrontierOptimizer(utilities[i])
		mv.optimize(gamble_samples[:,COMMUNITY_MEMBERSHIP[i]]-1)
		initial_portfolios[i][COMMUNITY_MEMBERSHIP[i]] = mv.weights
	return initial_portfolios