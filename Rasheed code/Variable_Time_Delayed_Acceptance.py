import numpy as np
from Variable_Time_Solver import *
from Functions_Library import likelihood, prior, stretch_mesh
import time

def var_delayed_acceptance(expensive_data,
                        cheap_data,
                        t_list,
                        alpha_0,
                        sigma_p,
                        sigma_l,
                        num_itr,
                        nx,
                        ny,
                        cheap_nx,
                        cheap_ny):
    '''
    A function that implements the delayed acceptance algorithm with variable time steps

    Arguments:

        expensive_data:       high resolution noise data 
        cheap_data:           low resolution noise data
        t_list:               a list of time steps 
        alpha_0:              initial value of the estimated parameter
        sigma_p:              variance of the proposal
        sigma_l:              variance of the likelihood function
        num_itr:              number of iterations
        nx, ny:               the number of triangles (expensive likelihood)
        cheap_nx, cheap_ny:   the number of triangles (cheap likelihood)
    '''
    start_time = time.time()   

    alpha = [alpha_0]   # initialize the markov chain

    # initialize a list to monitor acceptance ratio
    accepted_state_list = [alpha_0]
    acceptance_ratio_list = []

    
    # initialize the mesh, function space and source term
    mesh, V, u, v = stretch_mesh(nx=nx, ny=ny)
    cheap_mesh, cheap_V, cheap_u, cheap_v = stretch_mesh(nx=cheap_nx, ny=cheap_ny)

    A_0 = var_pred_exponential_time_dg(alpha_0, t_list, mesh, V, u, v)
    cheap_A_0 = var_pred_exponential_time_dg(alpha_0, t_list, cheap_mesh, cheap_V, cheap_u, cheap_v)

    i = 0
    
    while num_itr > i:

        alpha_prime = alpha_0 + sigma_p*np.random.normal(0, 1)   # propose a sample
        
        cheap_A = var_pred_exponential_time_dg(alpha_prime, t_list, cheap_mesh, cheap_V, cheap_u, cheap_v)

        # calculates the accetance ratio for the first rejection criterion
        acceptance_ratio = (likelihood(cheap_A, cheap_data, sigma_l) * prior(alpha_prime)) / (likelihood(cheap_A_0, cheap_data, sigma_l) * prior(alpha_0))
        
        # calculates the acceptance probability
        g = min(1, acceptance_ratio)

        if np.random.uniform(0,1) < g:   # will only proceed to the next stage if the current proposal is different from the previous one

            A = var_pred_exponential_time_dg(alpha_prime, t_list, mesh, V, u, v)
            
            # calculates the acceptance ratio for the second rejection criterion
            acceptance_ratio = (likelihood(A, expensive_data, sigma_l) * prior(alpha_prime)) / (likelihood(A_0, expensive_data, sigma_l) * prior(alpha_0)) 
            rho = min(1, acceptance_ratio)  

            A_0 = A

            # acceptance decision
            if np.random.uniform(0, 1) < rho:
                accepted_state_list.append(alpha_prime)
                alpha_0 = alpha_prime

        alpha.append(alpha_0)
        
        # update the acceptance ratio list 
        acceptance_ratio_list.append(len(accepted_state_list)/len(alpha))

        cheap_A_0 = cheap_A
        i += 1

    end_time = time.time()
    print('Computation time (mins): ', round((end_time - start_time)/60, 2))
    
    # save the chain and the accepted_ratio list 
    np.save('var_delayed_MCMC.npy', alpha)
    np.save('var_delayed_MCMC_accept_ratio.npy', acceptance_ratio_list)

    return alpha