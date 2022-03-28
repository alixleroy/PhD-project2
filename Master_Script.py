'''
Script that complies all four sampling techniques into one function for testing.
'''

from fenics import *
import numpy as np
from vedo.dolfin import plot, Latex, clear, histogram
import matplotlib.pyplot as plt
import pickle 
import time

from Master_Solver import master_solver
from Stretch_Mesh import stretch_mesh
from Time_Stepping import time_organiser

from Master_RWMH import *
from Master_Delayed_Acceptance import *
from Master_Irreversible_RWMH import *
from Master_WLB import *


def main():

    # Solver parameters.
    alpha0 = 1                              # Initial guess for alpha.
    tau = 1/10                              # Rate of growth of hot wall.
    epsilon = 1/100                         # Diffusion coefficient.
    num_steps = 60                          # Number of steps to run solver for.
    nx = 32                                 # Number of horizontal grid points on mesh (ny = nx).
    ny = 32                                 # Number of vertical grid points on mesh. 
    
    # Variable time-stepping variables.                                                                                                          
    dt_min = 1e-3                           # Inital value for dt.
    dt_max = 0.1                            # Limit of dt values as t appraches infty.
    reg = 10                                # Regularisation constant.

    # RWMH parameters.
    iterations = 2000                       # Number of samples to generate (-burnin).
    burnin = 0                              # Number of burn-in iterations.
    sigma_q = 0.1                           # Vairance of normal proposal distribution.
    sigma_p = 1                             # Variance of normal prior.
    mu_p = 0                                # Mean of normal prior.
    sigma_l = 0.2                           # Variance in likelihood (estimate of noise in data):

    # Delayed accpetance parameters.
    coarsening = 16                         # nx used in cheap likelihood approximation.
    num_steps_coarse = 60                   # Number of timesteps evaluated in likelihood approximation.

    # Irreversible RWMH varibles.
    alpha = 0.5 * 2/np.pi                         # alpha parameter used in gamma proposal (different to wind-speed variable).
    beta = 0.1 * np.sqrt(2/np.pi)                 # beta parameter used in gamma proposal.

    # Changing form of irreversilbe variables for use in in-built functions.
    k = alpha                               # 'Shape' parameter for gamma distribution (=alpha).
    theta = 1/beta                          # 'Scale' paramter for gamma distribution (=1/beta).

    # WLB variables.
    n = 100                                 # Number of i.d.d. data sets.
    tol = 1e-5                              # Tolerance of optimiser.
    bracket = (-5,5)                        # Bracket used in Brent scalar optimisation.

    # Data augmentation.
    # alpha_star = 0                        # True value for alpha.                          
    var_noise = 0.2                         # Variance of noise added to data (sigma_l should approxiamte this).

    # Open high resolution data file.
    with open(f'Master/Noisy_Data/INTERPOLATION_data_32_THIS_ONE!!', 'rb') as file:
        y = pickle.load(file)
    
    # Open low resolution data file.
    with open(f'Master/Noisy_Data/INTERPOLATION_COARSE_FINAL_data_16_interpolated_4', 'rb') as file:
        y_coarse = pickle.load(file)
    
    # # Thin high resolution data to correspond with variable time-stepping.
    # y_star_thinned = time_organiser(y_star,
    #                                 dt_min,
    #                                 dt_max,
    #                                 tau,
    #                                 reg,
    #                                 num_steps
    #                                 )[:-1] # Make removing last element more elegant!
    
    # # Thin low resolution data to correspond with variable time-stepping.
    # y_coarse_star_thinned = time_organiser(y_coarse_star,
    #                                        dt_min,
    #                                        dt_max,
    #                                        tau,
    #                                        reg,
    #                                        num_steps_coarse
    #                                        )[:-1] # Make removing last element more elegant!

    # # Generate vector of normally distributed 'noise' and add to data.
    # noise = np.random.normal(0, var_noise, np.shape(y_star_thinned))
    # y = y_star_thinned + noise
    # y_coarse = y_coarse_star_thinned + noise[:len(y_coarse_star_thinned),:len(y_coarse_star[0])] # This could be better!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    # start_RWMH = time.time()
    # # Run RWMH algorithm.
    # alpha_list_RWMH, __, ratio_list_RWMH = random_walk_metropolis(alpha0,
    #                                                               y, 
    #                                                               sigma_q, 
    #                                                               sigma_p,
    #                                                               sigma_l,
    #                                                               mu_p,
    #                                                               nx,
    #                                                               ny,
    #                                                               tau,
    #                                                               epsilon,
    #                                                               iterations,
    #                                                               burnin,
    #                                                               num_steps,
    #                                                               dt_min,
    #                                                               dt_max,
    #                                                               reg
    #                                                               )
    # end_RWMH = time.time()

    # time_RWMH = start_RWMH - end_RWMH

    # with open(f'/home/s1634937/Documents/Project 2/PhD-project2/Master/Sleepy_Data_2/alpha_list_RWMH_time{time_RWMH}', 'wb') as file:
    #     pickle.dump(alpha_list_RWMH, file)
    # with open(f'/home/s1634937/Documents/Project 2/PhD-project2/Master/Sleepy_Data_2/ratio_list_RWMH', 'wb') as file:
    #     pickle.dump(ratio_list_RWMH, file)

    # start_DA = time.time()
    # # Run delayed acceptance algortihm.
    # alpha_list_DA, __, ratio_list_DA, reject_first_list_DA = delayed_acceptance(alpha0,
    #                                                                          y,
    #                                                                          y_coarse,
    #                                                                          coarsening, 
    #                                                                          sigma_q, 
    #                                                                          sigma_p,
    #                                                                          sigma_l,
    #                                                                          mu_p,
    #                                                                          nx,
    #                                                                          tau,
    #                                                                          epsilon,
    #                                                                          iterations,
    #                                                                          burnin,
    #                                                                          num_steps,
    #                                                                          num_steps_coarse,
    #                                                                          dt_min,
    #                                                                          dt_max,
    #                                                                          reg
    #                                                                          )
    # end_DA = time.time()                                                                            
    
    # time_DA = start_DA - end_DA

    # with open(f'/home/s1634937/Documents/Project 2/PhD-project2/Master/Sleepy_Data_2/alpha_list_DA_time{time_DA}', 'wb') as file:
    #     pickle.dump(alpha_list_DA, file)
    # with open(f'/home/s1634937/Documents/Project 2/PhD-project2/Master/Sleepy_Data_2/ratio_list_DA', 'wb') as file:
    #     pickle.dump(ratio_list_DA, file)
    # with open(f'/home/s1634937/Documents/Project 2/PhD-project2/Master/Sleepy_Data_2/reject_first_list_DA', 'wb') as file:
    #     pickle.dump(reject_first_list_DA, file)

    start_IRWMH = time.time()
    # Run Irreversible RWMH algorithm.
    alpha_list_IRWMH, __, ratio_list_IRWMH = irreversible_walk_metropolis(alpha0,
                                                                          y, 
                                                                          k,
                                                                          theta, 
                                                                          sigma_p,
                                                                          sigma_l,
                                                                          mu_p,
                                                                          nx,
                                                                          ny,
                                                                          tau,
                                                                          epsilon,
                                                                          iterations,
                                                                          burnin,
                                                                          num_steps,
                                                                          dt_min,
                                                                          dt_max,
                                                                          reg
                                                                          )
    end_IRWMH = time.time()

    time_IRWMH = end_IRWMH - start_IRWMH                                                                          
    
    with open(f'Master/Sleepy_Data_3/alpha_list_IRWMH_time{time_IRWMH}', 'wb') as file:
        pickle.dump(alpha_list_IRWMH, file)
    with open(f'Master/Sleepy_Data_3/ratio_list_IRWMH', 'wb') as file:
        pickle.dump(ratio_list_IRWMH, file)
    
    print(alpha_list_IRWMH)
    print(ratio_list_IRWMH)
    # y_list = []
    # for _ in range(n):
    #     y_ = y_star_thinned + np.random.normal(0, var_noise, np.shape(y_star_thinned))
    #     y_list.append(y_.copy())

    # # Run WLB algorithm.
    # alpha_list_WLB = bootstrapping(iterations,
    #                                y_list,
    #                                sigma_l,
    #                                # alpha_0,
    #                                nx,
    #                                ny,
    #                                tau,
    #                                epsilon,
    #                                num_steps,
    #                                dt_min,
    #                                dt_max,
    #                                reg,
    #                                tol,
    #                                bracket
    #                                )

    # with open(f'/home/s1634937/Documents/Project 2/PhD-project2/Master/Sleepy_Data/alpha_list_WLB', 'wb') as file:
    #     pickle.dump(alpha_list_WLB, file)
    
    # with open(f'/home/s1634937/Documents/Project 2/PhD-project2/Master/Sleepy_Data/y_noisy', 'wb') as file:
    #     pickle.dump(y, file)

    return alpha_list_RWMH, ratio_list_RWMH, time_RWMH, alpha_list_DA, ratio_list_DA, time_DA, alpha_list_IRWMH, ratio_list_IRWMH, time_IRWMH

alpha_list_RWMH, ratio_list_RWMH, time_RWMH, alpha_list_DA, ratio_list_DA, time_DA, alpha_list_IRWMH, ratio_list_IRWMH, time_IRWMH = main()

print(alpha_list_RWMH)
print(ratio_list_RWMH)
print(alpha_list_DA)
print(ratio_list_DA)
print(alpha_list_IRWMH)
print(ratio_list_IRWMH)
print(time_RWMH)
print(time_DA)
print(time_IRWMH)
