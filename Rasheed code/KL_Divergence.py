import numpy as np

def estimate_distribution(data,
                        t_list,
                        alpha_list,
                        sigma_l,
                        nx,
                        ny):

    # initialize the mesh, function space
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

    # the relative entropy from q to p
    kl = np.sum(p * np.log2(p / q))

    return kl

