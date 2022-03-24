import numpy as np
from Variable_Time_Solver import *
from Functions_Library import gamma_proposal, proposal, likelihood, prior, stretch_mesh
import time


def var_gamma_non_reversible(t_list,
                        data,
                        nx,
                        ny,
                        alpha_0,
                        sigma_l,
                        k,
                        theta,
                        num_itr):
    
    '''
    A function that implements the non-reversible MCMC algorithm with variable time steps and gamma proposals

    Arguments:

        t_list:               a list of time steps 
        data:                 noise data 
        nx, ny:               the number of triangles
        alpha_0:              initial value of the estimated parameter
        sigma_l:              variance of the likelihood function
        k, theta:           parameters of the first proposal
        num_itr:              number of iterations
    '''
    
    start_time = time.time()   

    alpha = [alpha_0]   # initialize the markov chain

    # initialize a list to monitor acceptance ratio
    accepted_state_list = [alpha_0]
    acceptance_ratio_list = []

    
    # initialize the mesh, function space
    mesh, V, u, v = stretch_mesh(nx=nx, ny=ny)
    A_0 = var_pred_exponential_time_dg(alpha_0, t_list, mesh, V, u, v)
    
    i = 0
    zp = np.random.choice([1, -1], size=1)  # variable to change direction of the proposal
    zp = zp[0]
    
    while num_itr > i:
        
        u = np.random.uniform(0, 1)
        
        alpha_prime = alpha_0 + zp*np.random.gamma(k, theta)
        A = var_pred_exponential_time_dg(alpha_prime, t_list, mesh, V, u, v)

        # calculates the accetance ratio for the first proposal
        acceptance_ratio = (likelihood(A, data, sigma_l) * prior(alpha_prime)*gamma_proposal(alpha_0, k, theta)) / \
                               (likelihood(A_0, data, sigma_l) * prior(alpha_0)*gamma_proposal(alpha_prime, k, theta))

        # acceptance decision
        accept_prob = min(1, acceptance_ratio)
        if u < accept_prob:
            accepted_state_list.append(alpha_prime)
            alpha_0 = alpha_prime
            A_0 = A
        else:
            zp = -zp
        
        # update alpha list
        alpha.append(alpha_0)

        print(zp)

        # update the acceptance ratio list 
        acceptance_ratio_list.append(len(accepted_state_list)/len(alpha))

        i += 1

    end_time = time.time()
    print('Computation time (mins): ', round((end_time - start_time)/60, 2))
    
    # save the chain and the accepted_ratio list 
    np.save('var_gamma_nonreversible_MCMC.npy', alpha)
    np.save('var_gamma_nonreversible_MCMC_accept_ratio.npy', acceptance_ratio_list)

    return alpha
