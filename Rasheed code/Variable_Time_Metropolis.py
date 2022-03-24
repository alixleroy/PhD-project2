import numpy as np
from Variable_Time_Solver import *
from Functions_Library import likelihood, prior, stretch_mesh
import time

def var_metropolis(data,
                t_list,
                alpha_0,
                sigma_p,
                sigma_l,
                num_itr,
                nx,
                ny):
    '''
    A function that implements Metropolis algorithm with variable time steps

    Arguments:

        data:       noise data 
        t_list:     a list of time steps 
        alpha_0:    initial value of the estimated parameter
        sigma_p:    variance of the proposal
        sigma_l:    variance of the likelihood function
        num_itr:    number of iterations
        nx, ny:     the number of triangles
    '''
    start_time = time.time()  # start time count 

    alpha = [alpha_0]   # initialize the markov chain

    # initialize a list to monitor acceptance ratio
    accepted_state_list = [alpha_0]
    acceptance_ratio_list = []

    
    # initialize the mesh, function space and source term
    mesh, V, u, v = stretch_mesh(nx=nx, ny=ny)
    A_0 = var_pred_exponential_time_dg(alpha_0, t_list, mesh, V, u, v)
    
    for i in range(num_itr):

        alpha_prime = alpha_0 + sigma_p*np.random.normal(0, 1)   # random walk proposal

        A = var_pred_exponential_time_dg(alpha_prime, t_list, mesh, V, u, v)   # call the solver on generated alpha
        
        # calculates the acceptance ratio
        acceptance_ratio = (likelihood(A, data, sigma_l) * prior(alpha_prime)) / (likelihood(A_0, data, sigma_l) * prior(alpha_0)) 
        rho = min(1, acceptance_ratio) 

        # acceptance decision
        if np.random.uniform(0, 1) < rho:
            accepted_state_list.append(alpha_prime)
            alpha_0 = alpha_prime
            A_0 = A
            
        # update the list of alpha
        alpha.append(alpha_0)
        
        # update the acceptance ratio list 
        acceptance_ratio_list.append(len(accepted_state_list)/len(alpha))

    end_time = time.time()   # end time count
    print('Computation time (mins): ', round((end_time - start_time)/60, 2))
    
    # save the chain and the accepted_ratio list 
    np.save('var_metropolis.npy', alpha)
    np.save('var_metropolis_accept_ratio.npy', acceptance_ratio_list)

    return alpha    
                