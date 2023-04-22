import numpy as np
from tqdm import tqdm

import scipy.integrate as integrate
from scipy.optimize import minimize

#################################################################################################

# UTILITY FUNCTIONS
def u_x(x, gamma=2.1, A=1.2):
	"""
	Utility function for consumption of agent.
	"""
	return A*x**(1-gamma) / (1-gamma)

def v_x(x, gamma=2.1):
	"""
	Utility function for bequest of agent to offspring.
	"""
	return x**(1-gamma) / (1-gamma)

#################################################################################################

# SAFE STRATEGY 
def bequest_safe(w, c, r_hat):
	"""
	Bequest when using safe strategy.
	"""
	return (w-c)*r_hat

def consumption_utility(s, w, r_hat):
	"""
	Optimization function for safe investment strategy.
	We seek optimal s value.
	"""
	return -(u_x(w-s) + v_x(s*r_hat))

def no_project_best_utility(w, r_hat):
	"""
	Finds optimal s for safe investment strategy.
	Args:
		w: wealth
	Returns:
		optimal utility, optimal s value
	"""
	# optimization parameter is s, where 0 <= s <= w
	result = minimize(consumption_utility, args=(w, r_hat), bounds=((0,w),), x0=w/2, method="Powell")
	return -result.fun, result.x[0]

#################################################################################################

# RISKY STRATEGY
def bequest_project(s, beta, r, r_hat, r_bar, I):
	return (s - I)*r_hat + (1-beta)*I*r_bar + beta*I*r

def B_r(s, beta, r, r_hat, r_bar, I):
	return s*r_hat + I*(r_bar-r_hat) + I*beta*(r-r_bar)

def B_0(s, beta, r0, r_hat, r_bar, I):
	 return s*r_hat + I*(r_bar-r_hat) + I*beta*(r0-r_bar)

def optimal_beta_integral(s, beta, r0, r1, r_bar, r_hat, e_bar, I):
	return (1/(r1-r0)) * integrate.quad(lambda r : v_x(B_r(s, beta, r, r_hat, r_bar, I)), r0, r1)[0] - v_x(B_0(s, beta, r0, r_hat, r_bar, I)) - e_bar

def find_optimal_beta(s, r0, r1, r_bar, r_hat, e_bar, I, lower=0, upper=1, tolerance=1e-9, max_iters=80):
	"""
	Binary search to find zero of optimal_beta_integral function.
	"""
	iteration = 0
	while True:
		mid = (lower+upper)/2
		try:
			result = optimal_beta_integral(s, mid, r0, r1, r_bar, r_hat, e_bar, I)
		except:
			upper = 0.5
			continue
		if abs(result) < tolerance:
			break
		if result < 0:
			lower = mid
		elif result > 0:
			upper = mid
		iteration += 1
		if iteration > max_iters:
			return -1
	return mid

def project_optimization(s, w, r0, r1, r_bar, r_hat, e_bar, I):
	beta = find_optimal_beta(float(s), r0, r1, r_bar, r_hat, e_bar, I)
	B0 = B_0(s, beta, r0, r_hat, r_bar, I)
	return -(u_x(w-s) + v_x(B0))

def find_upper_s_bound(r0, r1, r_bar, r_hat, e_bar, I, lower=0, upper=5, tolerance=1e-9):
	while True:
		mid = (lower+upper)/2
		b = find_optimal_beta(mid, r0, r1, r_bar, r_hat, e_bar, I)
		if abs(b-1) < tolerance:
			return mid
		elif b == -1:
			upper = mid
		else:
			lower = mid

def project_best_utility(w, r0, r1, r_bar, r_hat, e_bar, I, upper_s_bound):
	"""
	Minimize project_optimization by finding optimal s.
	Optimal beta is only defined for s in (0.0, upper_s_bound), 
	so we set the bounds to (0, min(w, upper_s_bound)) since s <= w.
	"""
	# optimization parameter is s
	result = minimize(project_optimization, args=(w, r0, r1, r_bar, r_hat, e_bar, I), bounds=((0,min(w, upper_s_bound)),), x0 = w/2)
	s = result.x[0]
	utility = result.fun[0] 
	return -utility, s, find_optimal_beta(float(s), r0, r1, r_bar, r_hat, e_bar, I)

#################################################################################################

# SIMULATIONS
def simulate_lineage_wealth(w_init, r0, r1, r_hat, e_bar, I, reps):

	# compute other important constants
	r_bar  = (1/(r1-r0)) * integrate.quad(lambda x: x, r0, r1)[0] # expected project return
	upper_s_bound = find_upper_s_bound(r0, r1, r_bar, r_hat, e_bar, I)

	random_vals = np.random.uniform(r0, r1, size=reps)
	lineage_wealth = np.zeros((reps,))
	w = w_init

	# speedup
	max_wealth_project    = 0
	min_wealth_no_project = np.inf

	for rep_idx in tqdm(range(reps)):

		# speedup
		if w <= max_wealth_project:
			_, s_project, beta = project_best_utility(w, r0, r1, r_bar, r_hat, e_bar, I, upper_s_bound)
			w = bequest_project(s_project, beta, random_vals[rep_idx], r_hat, r_bar, I)
		elif w >= min_wealth_no_project:
			_, s_no_project = no_project_best_utility(w, r_hat)
			c = w - s_no_project
			w = bequest_safe(w, c, r_hat)

		else:
			utility_no_project, s_no_project = no_project_best_utility(w, r_hat)
			utility_project, s_project, beta = project_best_utility(w, r0, r1, r_bar, r_hat, e_bar, I, upper_s_bound)
			# choose project
			if utility_project > utility_no_project:
				max_wealth_project = max(w, max_wealth_project)
				w = bequest_project(s_project, beta, random_vals[rep_idx], r_hat, r_bar, I)
			# choose safe strategy
			else:
				min_wealth_no_project = min(w, min_wealth_no_project)
				c = w - s_no_project
				w = bequest_safe(w, c, r_hat)

		lineage_wealth[rep_idx] = w

	return lineage_wealth