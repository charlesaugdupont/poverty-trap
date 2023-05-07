import numpy as np
from tqdm import tqdm

import scipy.integrate as integrate
from scipy.optimize import minimize, root_scalar

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

# SAFE STRATEGY 
def bequest_safe(s, r_hat):
	"""
	Bequest when using safe strategy.
	"""
	return s*r_hat

def consumption_utility(s, w, r_hat):
	"""
	Optimization function for safe investment strategy.
	We seek optimal s value.
	"""
	return -(U(w-s) + V(s*r_hat))

def no_project_best_utility(w, r_hat):
	"""
	Finds optimal s for safe investment strategy.
	Args:
		w: wealth
	Returns:
		optimal utility, optimal s value
	"""
	# optimization parameter is s, where 0 <= s <= w
	result = minimize(consumption_utility, args=(w, r_hat), bounds=((w*0.1,w*0.9),), x0=w/2, method="Powell")
	return -result.fun, result.x[0]

#################################################################################################

# RISKY STRATEGY
def bequest_project(s, beta, r, r_hat, r_bar, I):
	"""
    Computes realized value of bequest for risky strategy with return r.
    Args:
        s    : amount saved
        beta : optimal beta
        r    : realized value of random variable for risky return
    Returns:
        the bequest amount
    """
	return (s - I)*r_hat + (1-beta)*I*r_bar + beta*I*r

def optimal_beta_integral(beta, s, r0, r1, r_bar, r_hat, e_bar, I):
	"""
    See Proposition 5.1
    Args:
        beta : beta value
        s    : amount saved
    Returns:
        IC'
    """
	return (1/(r1-r0)) * integrate.quad(lambda r : V(bequest_project(s, beta, r, r_hat, r_bar, I)), r0, r1)[0] - \
												   V(bequest_project(s, beta, r0, r_hat, r_bar, I)) - e_bar

def find_optimal_beta(s, r0, r1, r_bar, r_hat, e_bar, I):
	"""
    Function to find zero of optimal_beta_integral function, which corresponds to optimal beta.
    Args:
        s : amount saved
    Returns:
        optimal beta value; -1 if it does not exist
    """
	s = float(s)
	upper = 1.3 * (r_bar-r_hat) / (r_bar-r0)
	while True:
		try:
			result = root_scalar(optimal_beta_integral, args=(s, r0, r1, r_bar, r_hat, e_bar, I), bracket=(0,upper)).root
			assert not isinstance(result, complex)
			return result
		except:
			upper -= 0.05
			if upper < 0:
				return -1
			else:
				continue

def project_optimization(s, w, r0, r1, r_bar, r_hat, e_bar, I):
	"""
    Optimization function for the risky investment strategy.
    Args:
        s: amount saved
        w: wealth
    Returns:
        utility
    """
	beta = find_optimal_beta(s, r0, r1, r_bar, r_hat, e_bar, I)
	B0 = bequest_project(s, beta, r0, r_hat, r_bar, I)
	return -(U(w-s) + V(B0))

def project_best_utility(w, r0, r1, r_bar, r_hat, e_bar, I):
	"""
    Minimize project_optimization by finding optimal s.
    Args:
        w: wealth 
    Returns:
        optimal utility, optimal s, optimal beta
    """
	# optimization parameter is s
	result = minimize(project_optimization, args=(w, r0, r1, r_bar, r_hat, e_bar, I), bounds=((0,w),), method="SLSQP", x0=w/2)
	s = result.x[0]
	utility = result.fun[0] 
	return -utility, s, find_optimal_beta(s, r0, r1, r_bar, r_hat, e_bar, I)

#################################################################################################

# SIMULATIONS
def simulate_lineage_wealth(w_init, r0, r1, r_bar, r_hat, e_bar, I, reps):
	"""
	Simulate a lineage's wealth for reps periods.
	Args:
		w_init : initial wealth
		r0     : left bound of uniform random project return
		r1     : right bound ''
		r_bar  : expected project return
		r_hat  : safe return
		e_bar  : entrepreneurial effort needed to undertake project
		I      : capital needed to kickstart a project
		reps   : number of periods to run simulation for
	Returns:
		list of lineage wealth values over time
	"""
	random_vals = np.random.uniform(r0, r1, size=reps)
	lineage_wealth = np.zeros((reps,))
	w = w_init

	# speedup
	max_wealth_project    = 0
	min_wealth_no_project = np.inf

	for rep_idx in tqdm(range(reps)):

		# speedup
		if w <= max_wealth_project:
			_, s_project, beta = project_best_utility(w, r0, r1, r_bar, r_hat, e_bar, I)
			w = bequest_project(s_project, beta, random_vals[rep_idx], r_hat, r_bar, I)
		elif w >= min_wealth_no_project:
			_, s_no_project = no_project_best_utility(w, r_hat)
			w = bequest_safe(s_no_project, r_hat)

		else:
			utility_no_project, s_no_project = no_project_best_utility(w, r_hat)
			utility_project, s_project, beta = project_best_utility(w, r0, r1, r_bar, r_hat, e_bar, I)
			# choose project
			if beta != -1 and utility_project > utility_no_project:
				max_wealth_project = max(w, max_wealth_project)
				w = bequest_project(s_project, beta, random_vals[rep_idx], r_hat, r_bar, I)
			# choose safe strategy
			else:
				min_wealth_no_project = min(w, min_wealth_no_project)
				w = bequest_safe(s_no_project, r_hat)

		lineage_wealth[rep_idx] = w

	return lineage_wealth