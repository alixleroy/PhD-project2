import numpy as np
import typing
from typing import Callable, Tuple, List, Union
from solver_alix import solver_para, solver_loop
import time
from fenics import *
import numpy as np
from vedo.dolfin import plot, Latex, clear, histogram
import matplotlib.pyplot as plt
import pickle 

## Write a function to find the index of nearest element

def log_ratio(y: np.array, u1: np.array, u2:np.array, alpha1: float, alpha2: float, sigma_p: float, sigma_l: float, mu_p: float) -> np.array:
    '''
    
    '''
    return 0.5 * ((((alpha1 - mu_p) ** 2 - (alpha2 - mu_p) ** 2) / sigma_p ** 2) + (np.linalg.norm(y-u1) ** 2 - np.linalg.norm(y-u2) ** 2) / sigma_l ** 2)

def random_walk_metropolis(y: np.array, alpha0: float, iterations: int, sigma_q: float, sigma_p: float, sigma_l: float, mu_p: float, tau: float, epsilon: float, num_steps: int, T: float, nx: int, ny: int) -> np.array:
    '''

    '''

    #Empyt list to save vectors of importance 
    alpha_list = [alpha0]

    prop_list = []

    A_list = []

    K_list = []

    alpha1 = alpha0 #initialise the first steps 

    # Compute the solutions u with alpha = alpha 0 
    mesh, V, u, v, f = solver_para(nx_star,ny_star)
    u1, unorm1, tlist1 =solver_loop(alpha0,mesh,V,u,v,f,tau, epsilon, num_steps, T,w)
    
    for i in range(iterations):


        alpha2 = np.random.normal(alpha1, sigma_q)

        u2, unorm2, tlist2=solver_loop(alpha2,mesh,V,u,v,f,tau, epsilon, num_steps, T,w)

        A = log_ratio(y, unorm1, unorm2, alpha1, alpha2, sigma_p, sigma_l, mu_p)

        K = np.log(np.random.uniform(0,1))

        if A >= 0 or K <= A:

            alpha1 = alpha2

        alpha_list.append(alpha1)

        u1 = u2

        prop_list.append(alpha2)

        A_list.append(A)

        K_list.append(K)


    return alpha_list, prop_list, A_list, K_list




## Run the metropolis hasting algorithm

alpha0 = 1 # Initial value of alpha - start at something else
iterations = 1000 # Lenght of the MCMC chain
tau =1/10 #Value of tau
epsilon = 1/20 #
num_steps = 10 #number of time steps taken  ########## VERIFY WITH ELLIOT ##########
T = 5.0 #final time 
nx = 36 #size of the grid
ny = 36 #size of the grid
sigma_q = 0.1 #variance of the guess proposal
sigma_p = 1 #variance of the prior
mu_p = 0 #mean of the proposal 
sigma_l = 0.2 #variance of the likelihood
w = Expression(('2*x[1]*(1-x[0]*x[0])', '-2*x[0]*(1-x[1]*x[1])'), degree=3) # wind expression

## Generate the observed data
var_noise = 0.3 # noise added to the data
alpha_star = 0
nx_star=36
ny_star=36
mesh, V, u, v, f = solver_para(nx_star,ny_star)
u_list, unorm_list, t_list=solver_loop(alpha_star,mesh,V,u,v,f,tau, epsilon, num_steps, T,w)
y_norm_star = unorm_list + np.random.normal(np.zeros(np.shape(unorm_list)), var_noise)


time1 = time.time()
alpha_list, prop_list, A_list, K_list = random_walk_metropolis(y_norm_star, alpha0, iterations, sigma_q, sigma_p, sigma_l, mu_p, tau, epsilon, num_steps, T, nx, ny)
time2 =time.time()-time1 
print(time2)


fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(18, 5)) #create the figure 
ax1.plot(alpha_list) #plot values of the element 0 of the chain 

bins = np.linspace(-3., 3., 100) #set bins to plot an histogram of the values of X^0
step = bins[1]-bins[0] #adjust the setp 
hist,bin_edges = np.histogram(alpha_list, bins=bins,density=True) # get the histogram
hist_norm = hist/(sum(hist)*step) #normalise the histogram
ax2.bar(bin_edges[:-1], hist_norm, width = step, alpha = 0.5) #plot the histogram

fig.savefig("Alixcode/markov_chain.png")

name_file = "Alixcode/markovchain_normtimeT/alpha_list-M="+str(iterations)+"-time = "+str(round(time2,1))+".csv"
with open(name_file, 'wb') as f:
    pickle.dump(alpha_list, f)

name_file = "Alixcode/markovchain_normtimeT/prop_list-M="+str(iterations)+"-time = "+str(round(time2,1))+".csv"
with open(name_file, 'wb') as f:
    pickle.dump(prop_list, f)

name_file = "Alixcode/markovchain_normtimeT/A_list-M="+str(iterations)+"-time = "+str(round(time2,1))+".csv"
with open(name_file, 'wb') as f:
    pickle.dump(A_list, f)

name_file = "Alixcode/markovchain_normtimeT/K_list-M="+str(iterations)+"-time = "+str(round(time2,1))+".csv"
with open(name_file, 'wb') as f:
    pickle.dump(K_list, f)