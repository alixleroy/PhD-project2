import fenics
import numpy as np
import matplotlib.pyplot as plt
import typing
from typing import Callable, Tuple, List, Union
import pickle
from solver_alix import solver_para, solver_run
import time

def log_ratio(y: np.array, u1: np.array, u2:np.array, alpha1: float, alpha2: float, sigma_p: float, sigma_l: float, mu_p: float) -> np.array:
    '''
    
    '''
    return 0.5 * ((((alpha1 - mu_p) ** 2 - (alpha2 - mu_p) ** 2) / sigma_p ** 2) + (np.linalg.norm(y - u1) ** 2 - np.linalg.norm(y - u2) ** 2) / sigma_l ** 2)


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
    Vmesh,mesh,u,v,u_n,f,u_D,bc= solver_para(nx,ny,tau)
    u1 = solver_run(alpha1,Vmesh,mesh,u,v,u_n,f,u_D,bc,tau, epsilon, num_steps, T)

    for i in range(iterations):


        alpha2 = np.random.normal(alpha1, sigma_q)

        u2 = solver_run(alpha2,Vmesh,mesh,u,v,u_n,f,u_D,bc,tau, epsilon, num_steps, T)

        A = log_ratio(y, u1, u2, alpha1, alpha2, sigma_p, sigma_l, mu_p)

        K = np.log(np.random.uniform(0,1))

        if A >= 0 or K <= A:

            alpha1 = alpha2

        alpha_list.append(alpha1)

        u1 = u2

        prop_list.append(alpha2)

        A_list.append(A)

        K_list.append(K)

        print(i)

    return alpha_list, prop_list, A_list, K_list




## Run the metropolis hasting algorithm

alpha0 = 1 # Initial value of alpha

iterations = 5000  # Lenght of the MCMC chain

tau =1 #Value of tau
epsilon = 1/20 #
num_steps = 100 #number of time steps taken
T = 5.0 #final time 
nx = 30 #size of the grid
ny = 30 #size of the grid

sigma_q = 0.5 #variance of the guess proposal
sigma_p = 1 #variance of the prior
mu_p = 0 #mean of the proposal 
sigma_l = 0.2 #variance of the likelihood
        

## Generate the observed data
noise_star = 1 #noise added to the data
alpha_star = 0
nx_star=30
ny_star=30 
Vmesh,mesh,u,v,u_n,f,u_D,bc= solver_para(nx_star,ny_star,tau)
u_star = solver_run(alpha_star,Vmesh,mesh,u,v,u_n,f,u_D,bc,tau, epsilon, num_steps, T)
y_star = u_star + np.random.normal(np.zeros(np.shape(u_star)), noise_star)

time1 = time.time()
alpha_list, prop_list, A_list, K_list = random_walk_metropolis(y_star, alpha0, iterations, sigma_q, sigma_p, sigma_l, mu_p, tau, epsilon, num_steps, T, nx, ny)
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

name_file = "Alixcode/alpha_res/chain-M="+str(iterations)+".csv"
with open(name_file, 'wb') as f:
    pickle.dump(alpha_list, f)