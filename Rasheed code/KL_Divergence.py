import numpy as np
from Variable_Time_Solver import *
from Functions_Library import load_data, likelihood, prior, stretch_mesh
from Time_stepping import *
import matplotlib.pyplot as plt
import seaborn as sns

def estimate_distribution(data,
                        t_list,
                        alpha_list,
                        sigma_l,
                        nx,
                        ny):

    # initialize the mesh, function space and source term
    mesh, V, u, v = stretch_mesh(nx=nx, ny=ny)
    estimation_list = []

    for alpha in alpha_list:

        A = var_pred_exponential_time_dg(alpha, t_list, mesh, V, u, v)
        
        lp = likelihood(A, data, sigma_l) * prior(alpha)
        estimation_list.append(lp)
    
    norm_constant = np.sum(estimation_list)
    
    estimation_list = np.array(estimation_list) / norm_constant

    return estimation_list

def alpha_probabilty(alpha_list,
                     nbins):

    n, bin_edges = np.histogram(alpha_list, bins=nbins)
    totalcount = np.sum(n)
    bin_probability = n / totalcount

    bins_per_point = np.fmin(np.digitize(alpha_list, bin_edges), len(bin_edges)-1)
    probability_perpoint = [bin_probability[bins_per_point[i]-1] for i in range(len(alpha_list))] 

    return probability_perpoint

def kl_divergence_measure(data,
                        t_list,
                        alpha_list,
                        sigma_l,
                        nx,
                        ny,
                        nbins):

    p = alpha_probabilty(alpha_list,
                     nbins)
    
    q = estimate_distribution(data,
                        t_list,
                        alpha_list,
                        sigma_l,
                        nx,
                        ny)

    # the relative entropy from p to q
    kl = np.sum(p * np.log2(p / q))

    return kl

# Load data 
data = load_data('master_data_32_project')

# Find t_list
t_list = find_t_list(dt_min=1e-7,
                dt_max=0.1,
                tau=1/10,
                reg=10,
                num_steps=60
                )

# Organize the data 
organized_data = time_organiser(y_data = data,
                   dt_min=1e-7,
                   dt_max=0.1,
                   tau=1/10,
                   reg=10,
                   num_steps=60,
                   data_res=0.00001
                   )

alpha_list = np.load('var_delayed_MCMC.npy')

kl = kl_divergence_measure(data = organized_data,
                        t_list = t_list,
                        alpha_list = alpha_list,
                        sigma_l = 0.2,
                        nx=32,
                        ny=32,
                        nbins=100)

print(kl)

'''

print(estimate_distribution(organized_data,
                    t_list,
                    alpha_list[1990:],
                    0.2,
                    32,
                    32)
)
'''