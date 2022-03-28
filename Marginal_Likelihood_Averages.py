'''
Script to numerically integrate the prior-likelihood function to obtain the marginal likelihood Z. Uses spatial averages variation of the likelihood function.
'''

import numpy as np

'''
Script to calculate the integral of the likelihood (Z) and log of the posterior numerically, for use in 
calculating the Kullback-Leibler Divergence.
'''
import numpy as np
from Stretch_Mesh import stretch_mesh
from Master_Solver import master_solver

def scaled_normal_value(alpha,
                        sigma_b,
                        scale
                        ):

    value = (1/(sigma_b * np.sqrt(2 * np.pi))) * np.exp(-(1/2) * (alpha/sigma_b) ** 2) * scale

    return value

def binner(start,
           stop,
           num_bins,
           sigma_b,
           scale
           ):
    
    mid = (stop - start)/2

    bins = np.linspace(-mid, mid, num_bins + 1)

    stretchy_bins = [-mid]

    for i in range(len(bins)-2):

        stretchy_bins.append(bins[i+1] / abs(scaled_normal_value(bins[i+1],
                                                                 sigma_b,
                                                                 scale
                                                                 )))
    
    stretchy_bins.append(mid)

    bins = (start + stop)/2 * np.ones(len(stretchy_bins)) + np.array(stretchy_bins)

    return bins


def A_mat(u):
    
    return u

def prior_likelihood(y,
                     alpha,
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
                     ):
    
    u, __ = master_solver(mesh,
                          tau,
                          alpha,
                          epsilon,
                          num_steps,
                          dt_min,
                          dt_max,
                          reg
                          )
    
    # Take average in space.
    u = np.mean(u, axis=1)

    PL = (1/(sigma_l * sigma_p * 2 * np.pi)) * np.exp(-0.5 * (((alpha - mu_p)/sigma_p) ** 2 + (np.linalg.norm(y - A_mat(u))/sigma_l) ** 2) + B)

    return PL

def integrate_prior_likelihood(y,
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
                               ):
    
    a = bins[0]

    PL_a = prior_likelihood(y,
                            a,
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

    PL_list = [PL_a]
    PL_int_list = []

    for i in range(len(bins)-1):
        
        print(i)

        b = bins[i+1]
        
        PL_b = prior_likelihood(y,
                                b,
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
        
        area_trapezium = ((PL_a + PL_b)/2) * (b - a)

        PL_list.append(PL_b.copy())
        PL_int_list.append(area_trapezium.copy())

        a = b
        PL_a = PL_b
    
    return PL_list, PL_int_list


if __name__ == "__main__":

    import pickle
    from Time_Stepping import time_organiser

    def main():

        # Solver parameters.
        tau = 1/10                              # Rate of growth of hot wall.
        epsilon = 1/100                         # Diffusion coefficient.
        num_steps = 60                          # Number of steps to run solver for.
        nx = 16                                 # Number of horizontal grid points on mesh (ny = nx).
        ny = 16                                 # Number of vertical grid points on mesh. 

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
        start = -3                              # Start-point of interval to integrate over.
        stop = 3                                # End-point of interval to integrate over.
        num_bins = 100                          # Number of bins to integrate over.
        sigma_b = np.sqrt(6)                    # Variance of normally streched bins.
        scale = 14                              # Scaling constant for stretchy bins.

        B = 0 # 3.28625e4                       # Pertubation constant to artificially scale function. Keep 0 here!

        # Generate stretched bins.
        bins = binner(start,
                      stop,
                      num_bins,
                      sigma_b,
                      scale
                      )

        # Generate stretchy mesh from Stretch_Mesh.py.
        mesh = stretch_mesh(nx=nx, ny=ny)

        # # Load data file.
        # with open('Master/Data_eps100_num150000_tau10_alpha0/master_data_32_interpolate', 'rb') as file:
        #     y_star = pickle.load(file)

        # # Format data to match variable time-stepping scheme.
        # y_star_coarse = time_organiser(y_star,
        #                                dt_min,
        #                                dt_max,
        #                                tau,
        #                                reg,
        #                                num_steps
        #                                )[:-1] # Make removing last element more elegant!

        # # Take mean in space.
        # y_star_coarse_av = np.mean(y_star_coarse, axis=1)

        # # Add normally distributed noise.
        # y = y_star_coarse_av + np.random.normal(0, var_noise, np.shape(y_star_coarse_av))

        # Load data file preformatted and with noise.
        with open('Master/Noisy_Data/data_32_thisistheoneweareusing!!!_it _noisy_averaged', 'rb') as file:
            y = pickle.load(file)

        # # Average data in space.
        # y = np.mean(y_unaveraged, axis=1)

        # Run function that integrates the likelihood.
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

        Z = np.sum(PL_int_list)

        # with open(f'Master/Noisy_Data/data_16_interpolated', 'wb') as file:
        #     pickle.dump(y, file, protocol=2)

        return PL_list, PL_int_list, bins, Z

    PL_list, PL_int_list, bins, Z = main()

    print('PL_list')
    print(PL_list)
    print('PL_int_list')
    print(PL_int_list)
    print('bins')
    print(list(bins))
    print(f'Z = {Z}')