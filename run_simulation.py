from model import *
import pickle

from SALib.sample import latin
from SALib.analyze import pawn


gamma  = 2.1 # utility function parameter
A      = 1.2 # utility function parameter
r0     = 0.5 # left bound for uniform random return on project
r1     = 2.5 # right bound for uniform random return on project
r_hat  = 1.1 # safe return
I      = 1.0 # amount needed to undertake project
e_bar  = 1.5 # amount of entrepreneurial effort needed to undertake project
r_bar  = (1/(r1-r0)) * integrate.quad(lambda x: x, r0, r1)[0] # expected project return
w_init = 0.1

# define problem
PROBLEM = {
	"num_vars" : 3,
	"names"    : ["I", "e_bar", "r1"],
	"bounds"   : [[0.8, 1.2],
				  [1.0, 1.7],
				  [2.0, 3.0]]
}

NUM_SAMPLES = 4

# LHS sampling
X = latin.sample(PROBLEM, NUM_SAMPLES)

results = {}

for idx, row in enumerate(X):
	I, e_bar, r1 = row
	I            = float(I)
	e_bar        = float(e_bar)
	r1           = float(r1)

	key = (I, e_bar, r1)
	results[key] = {}

	r_bar  = (1/(r1-r0)) * integrate.quad(lambda x: x, r0, r1)[0]
	upper_s_bound = find_upper_s_bound(r0, r1, r_bar, r_hat, e_bar, I)
	LW = simulate_lineage_wealth(w_init, r0, r1, r_bar,r_hat, e_bar, I, upper_s_bound, reps=5000)
	
	results[key]["trajectory"] = LW
	results[key]["mean"]       = np.mean(LW)
	results[key]["skew"]       = skew(LW)


with open("results.pickle", "wb") as f:
	pickle.dump(results, f)


# Si = pawn.analyze(PROBLEM, X, Y, S=10, print_to_console=True)