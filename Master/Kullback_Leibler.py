'''
Script to calculate the Kullback-Leibler divergence between an analytical distribution and a sampling distribution.
'''

import numpy as np

### Discrete ###

# def hist_value(alpha,
#                hist,
#                bins
#                ):

#     for i in range(len(bins)-1):
#         if bins[i]<alpha:
#             k = i-1
#             break
    
#     Q_approx = hist[k]

#     return Q_approx

# def KL_div_discrete(alpha_list,
#                     P
#                     ):
    
#     hist, bins = np.hist(alpha_list)

#     KL_sum = 0

#     for alpha in alpha_list:

#         P_alpha = P(alpha_list[i])
#         Q_approx = hist_value(alpha,
#                               hist,
#                               bins
#                               )

#         KL_sum += P_alpha * np.log(P_alpha/Q_approx)
    
#     return KL_sum

### Continuous Approximation ###

def KL_div(alpha_list,
           bins,
           sigma_l,
           sigma_p,
           mu_p,
           B,
           K,
           LL_int_list
           ):

    hist, __ = np.histogram(alpha_list, 
                            bins
                            )
    
    hist = hist / len(alpha_list)

    KLD = 0

    for i in range(len(bins)-1):
        
        a = bins[i]
        b = bins[i+1]
        P = hist[i]

        log_Q_int_i = (B - np.log(sigma_p * K * np.sqrt(2 * np.pi)) + mu_p/(2 * sigma_p ** 2)) * (b - a) - (1/4 * sigma_p ** 2) * (b ** 2 - a ** 2) - LL_int_list[i]

        # To avoid log(0) - Think about this!!!!!!!!!! @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@'
        if P != 0:
            KLD += P * np.log(P) * (b-a) - P * log_Q_int_i

    return KLD, hist
    

# Maybe we change this so instead of summing over alpha ina lpha_list we
# sum over the centre of each bin - maybe need to find a way to do integral?

# MAYBE INTEGRATE OVER BINS!!! APPROXIMATE INTEGRAL BY SPLITTING INTO BINS

if __name__ == "__main__":

    from Log_Likelihood_Integration import integrate_log_likelihood
    from Marginal_Likelihood import integrate_prior_likelihood, binner
    from Stretch_Mesh import stretch_mesh
    import pickle

    def main():

        # Solver parameters.
        tau = 1/10                              # Rate of growth of hot wall.
        epsilon = 1/100                         # Diffusion coefficient.
        num_steps = 60                          # Number of steps to run solver for.
        nx = 32                                 # Number of horizontal grid points on mesh (ny = nx).
        ny = 32                                 # Number of vertical grid points on mesh. 

        # Variable time-stepping variables.                                                                                                          
        dt_min = 1e-3                           # Inital value for dt.
        dt_max = 0.1                            # Limit of dt values as t appraches infty.
        reg = 10                                # Regularisation constant.

        # Bayesian parameters.                      
        sigma_p = 1                             # Variance of normal prior.
        mu_p = 0                                # Mean of normal prior.
        sigma_l = 0.2                           # Variance in likelihood (estimate of noise in data):

        # Data augmentation.
        # alpha_star = 0                        # True value for alpha.                          
        var_noise = 0.2                         # Variance of noise added to data (sigma_l should approxiamte this).

        # Integration Parameters.
        start = -1 # -2                         # Start-point of interval to integrate over.
        stop = 1 # 2                            # End-point of interval to integrate over.
        num_bins = 100                          # Number of bins to integrate over.
        sigma_b = np.sqrt(2)                    # Variance of normally streched bins.
        scale = 5 # 10                          # Scaling constant for stretchy bins.

        B = 3.28625e4                           # Pertubation constant to artificially scale prior-likelihood function.

        # Load the sampling distribution of interest.
        with open('Master/RWMH_Sampling_Data/5100iter_100burnin_sq01_sp1_mp0_sl02_noise02_nx32_tau01_eps001_numsteps60-150000_SAMPLE', 'rb') as file:
            alpha_list = pickle.load(file)

        # Load corresponding noisy data used.
        with open('Master/RWMH_Sampling_Data/5100iter_100burnin_sq01_sp1_mp0_sl02_noise02_nx32_tau01_eps001_numsteps60-150000_DATA', 'rb') as file:
            y = pickle.load(file)

        # Create same mesh used in sampling method.
        mesh = stretch_mesh(nx=nx, ny=ny)

        # Generate stretched bins to integrate over.
        bins = binner(start,
                      stop,
                      num_bins,
                      sigma_b,
                      scale
                      )

        # Numerically integrate scaled prior-likelihood function to approximate K = Z * e^B.
        PL_list, PL_int_list = integrate_prior_likelihood(y,
                                                          bins,
                                                          sigma_l,
                                                          sigma_p,
                                                          mu_p,
                                                          B,
                                                          mesh,
                                                          tau,
                                                          epsilon,
                                                          num_steps,
                                                          dt_min,
                                                          dt_max,
                                                          reg
                                                          )
        
        K = np.sum(PL_int_list)

        # Numerically integrate log likelihood function into bins.
        LL_list, LL_int_list = integrate_log_likelihood(y,
                                                        bins,
                                                        sigma_l,
                                                        # B,
                                                        mesh,
                                                        tau,
                                                        epsilon,
                                                        num_steps,
                                                        dt_min,
                                                        dt_max,
                                                        reg
                                                        )

        KLD, hist = KL_div(alpha_list,
                           bins,
                           sigma_l,
                           sigma_p,
                           mu_p,
                           B,
                           K,
                           LL_int_list
                           )

        return KLD, hist, bins, PL_list, PL_int_list, LL_list, K
    
    KLD, hist, bins, PL_list, PL_int_list, LL_list, K = main()

    print('KLD')
    print(KLD)
    print('hist')
    print(hist)
    print('bins')
    print(list(bins))
    print('PL_list')
    print(PL_list)
    print('PL_int_list')
    print(PL_int_list)
    print('LL_list')
    print(LL_list)
    print('K')
    print(K)