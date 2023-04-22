from model import *
import pickle


r0     = 0.5 # left bound for uniform random return on project
r1     = 2.5 # right bound for uniform random return on project
r_hat  = 1.1 # safe return
I      = 1.0 # amount needed to undertake project
e_bar  = 1.5 # amount of entrepreneurial effort needed to undertake project
w_init = 0.1 # initial wealth of agents


LW = simulate_lineage_wealth(w_init, r0, r1, r_hat, e_bar, I, reps=5000)

with open("./out.pickle", "wb") as f:
    pickle.dump(LW, f)