from fenics import *
import numpy as np
from vedo.dolfin import plot, Latex, clear, histogram
import matplotlib.pyplot as plt
import pickle 
import time

from Function_Solver import solver_para, solver_run

from random_walk_MCMC_integrate import *

alpha_star = 0
alpha0 = 1
iterations = 1000
tau = 1/10
epsilon = 1/20
num_steps = 100
nx = 36
ny = 36

sigma_q = 0.1
sigma_p = 1
mu_p = 0
sigma_l = 0.2

var_noise = 0.3

V, u, v, u_n, f, u_D, bc, mesh = solver_para(nx, ny, tau)
y_star = solver_run(alpha_star, V, u, v, u_n, f, u_D, bc, epsilon, num_steps)
# noise = np.random.normal(np.zeros((nx + 1)*(ny + 1)), var_noise)
# noise_function = Function(V)
# noise_function.vector()[:] = noise

for f in y_star:
    noise = np.random.normal(np.zeros((nx + 1)*(ny + 1)), var_noise)
    # noise_function = Function(V)
    f.vector()[:] += noise
    # y.append(f + noise_function)

y = y_star

for i in range(2, 42, 4):

    begin = time.time()
    alpha_list_i = random_walk_metropolis(alpha0, y, sigma_q, sigma_p, sigma_l, mu_p, nx, ny, tau, epsilon, iterations, num_steps, i)
    end = time.time()

    time_difference = end - begin

    name_file = "Elliot_Code/Function_Distributions/"+str(i)+"_steps_considered_time_"+str(time_difference)+".csv"
    with open(name_file, 'wb') as f:
        pickle.dump(alpha_list_i, f)