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
    '''
    Returns probability density of alpha with respect to a chosen normal distribution.
    '''
    value = (1/(sigma_b * np.sqrt(2 * np.pi))) * np.exp(-(1/2) * (alpha/sigma_b) ** 2) * scale

    return value


def binner(start,
           stop,
           num_bins,
           sigma_b,
           scale
           ):
    '''
    Partitions an interval into sub-intervals (bins) such that sub-intervals are smaller towards the center.

    Inputs:
        start:          Float:              Start of interval.
        stop:           Float:              End of interval.
        num_bins:       Int:                Number of bins to divide inteval into.
        sigma_b:        Float:              Variance for normally distributed bins.
        scale:          Float:              Scaling coefficient for bin distribution.
    Outputs:
        bins:           List:               List of start and end points for each bin.        
    '''
    
    # Find centre of interval.
    mid = (stop - start)/2

    # Partition interval of same length with centre 0 into bins.
    bins = np.linspace(-mid, mid, num_bins + 1)

    # Initialise streched bin list.
    stretchy_bins = [-mid]

    # Stretch bins according to given normal distribution.
    for i in range(len(bins)-2):
        stretchy_bins.append(bins[i+1] / abs(scaled_normal_value(bins[i+1],
                                                                 sigma_b,
                                                                 scale
                                                                 )))
    
    # Add end point to list.
    stretchy_bins.append(mid)

    # Recentre interval at original locaiton.
    bins = (start + stop)/2 * np.ones(len(stretchy_bins)) + np.array(stretchy_bins)

    return bins


def A_mat(u):
    '''
    Formats solver output into correct shape for the given data.
    '''
    return u


def log_likelihood(y,
                   u,
                   sigma_l
                   ):
    '''
    Evaluates log-likelihood at alpha value used to generate u.
    '''
    LL = np.log(2/(sigma_l * np.sqrt(2 * np.pi))) - (1/(2 * sigma_l ** 2)) * np.linalg.norm(y - A_mat(u)) ** 2

    return LL


def integrate_log_likelihood(y,
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
                             ):
    '''
    Function to integrate log-likelihood using the trapezium rule for given parameters.

    Inputs:
        y:                          np.Array:           Generated data encoding information about heat distribution in time.
        bins:                       List:               List of the start an end points for the trapeziums.
        sigma_l:                    Float:              Variance of the likelihood function.
        mesh:                       FEniCS mesh:        Mesh used in the FEniCS finite elements solver for the double glazing problem.
        tau:                        Float:              Rate of growth of hot wall boundary condition in oduble glazing problem.
        epsilon:                    Float:              Constant of diffusion in advection-diffusion equation.
        num_steps:                  Int:                Number of steps in time to run solver.
        dt_min:                     Float:              Initial dt for vairbale time-stepping scheme.
        dt_max:                     Float:              Limit of dt's in variable time-stepping scheme.
        reg:                        Float:              Regularisation constant for vairbale time-stepping scheme, chosen heuristically.
    Outputs:    
        log_likelihood_list:        List:               List of log-likelihoods evaultated at each point in bins.
        log_likelihood_int_list:    List:               List of log_likelihood integral aprroximations in each region between point in bins.  
    '''
    # Assign beginning of interval.
    a = bins[0]

    # Solve double glazing problem for alpha = a.
    u_a, __ = master_solver(mesh,
                            tau,
                            a,
                            epsilon,
                            num_steps,
                            dt_min,
                            dt_max,
                            reg
                            )

    # Take average in space.
    u_a = np.mean(u_a, axis=1)

    # Calculate initial log likelihood.
    log_likelihood_a = log_likelihood(y,
                                      u_a,
                                      sigma_l
                                      )

    # Initialise log likelikihood and log likelihood integral lists.
    log_likelihood_list = [log_likelihood_a]
    log_likelihood_int_list = []

    # Begin integration via trapezium rule.
    for i in range(len(bins)-1):
        
        # Sanity check.
        print(i)

        # Assign end of sub-interval.
        b = bins[i+1]

        # Run double glazing solver for alpha = b.
        u_b, __ = master_solver(mesh,
                                tau,
                                b,
                                epsilon,
                                num_steps,
                                dt_min,
                                dt_max,
                                reg
                                )

        # Take average in space.
        u_b = np.mean(u_b, axis=1)

        # Calculate log-likelihood for given b.
        log_likelihood_b = log_likelihood(y,
                                          u_b,
                                          sigma_l
                                          )
        
        # Caluclate corresponding aread of trapezium across bin.
        area_trapezium = ((log_likelihood_a + log_likelihood_b)/2) * (b - a)

        # Update lists.
        log_likelihood_list.append(log_likelihood_b.copy())
        log_likelihood_int_list.append(area_trapezium.copy())

        # Update variables.
        a = b
        log_likelihood_a = log_likelihood_b
    
    return log_likelihood_list, log_likelihood_int_list


if __name__ == "__main__":

    import pickle
    from Time_Stepping import time_organiser

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
        # sigma_p = 1                           # Variance of normal prior.
        # mu_p = 0                              # Mean of normal prior.
        sigma_l = 0.2                           # Variance in likelihood (estimate of noise in data):

        # Data augmentation.
        # alpha_star = 0                        # True value for alpha.                          
        var_noise = 0.2                         # Variance of noise added to data (sigma_l should approxiamte this).

        # Integration Parameters.
        start = -2                              # Start-point of interval to integrate over.
        stop = 2                                # End-point of interval to integrate over.
        num_bins = 100                          # Number of bins to integrate over.
        sigma_b = np.sqrt(2)                    # Variance of normally streched bins.
        scale = 10                              # Scaling constant for stretchy bins.
        B = 3.28625e4

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

        # # Add normally distibuted noise.
        # y = y_star_coarse + np.random.normal(0, var_noise, np.shape(y_star_coarse))

        # Load data file preformatted and with noise.
        with open('Master/Noisy_Data/data_32_thisistheoneweareusing!!!_it _noisy_averaged', 'rb') as file:
            y = pickle.load(file)

        # # Average data in space.
        # y = np.mean(y_unaveraged, axis=1)

        # Run function that integrates the likelihood.
        log_likelihood_list, log_likelihood_int_list = integrate_log_likelihood(y,
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

        return log_likelihood_list, log_likelihood_int_list

    log_likelihood_list, log_likelihood_int_list = main()

    print('log_likelihood_list')
    print(log_likelihood_list)
    print('log_likelihood_int_list')
    print(log_likelihood_int_list)