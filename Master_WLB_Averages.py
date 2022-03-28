'''
Implementation of weighted likelihood bootstrapping algorithm for the double-glazing inverse problem. Referece paper: "Introducing prior information in Weighted Likelihood Bootstrap
with applications to model misspecification" by Emily Pompe. Scalar version (might not be i.i.d.)
'''
import numpy as np
from scipy.optimize import minimize_scalar
from Stretch_Mesh import stretch_mesh

from Master_Solver import master_solver
from Time_Stepping import time_organiser

def A_mat(u):
    '''
    Format u.
    '''
    return u

def log_likelihood(u,
                   y_list,
                   sigma_l
                   ):
    '''
    Calcluates the log of the likelihood function.

    Inputs:
        u:                      List:       List representing point values of the numerical solution at each vertice in time.
        y_list:                 List:       List of y's with i.i.d. noise, where each y has the same shape as u and represents the observed data.
        sigma_l:                Float:      Variance in likelihood (estimate of the noise in the data).
    Outputs:
        log_likelihood_list:    List:       List of the logs of the likelihood evaluated for each dataset y.
    '''

    # Initialise list.
    log_likelihood_list = []

    # For each data set y, caluclate the log likelihood and append to list.
    for y in y_list:
        f = np.log(2/(sigma_l * np.sqrt(2 * np.pi))) - (1/(2 * sigma_l ** 2)) * np.linalg.norm(y - A_mat(u)) ** 2
        log_likelihood_list.append(f.copy())

    return np.array(log_likelihood_list)

def weighted_likelihood_neg(alpha,
                            w,
                            y_list,
                            sigma_l,
                            mesh,
                            tau,
                            epsilon,
                            num_steps,
                            dt_min,
                            dt_max,
                            reg
                            ):
    '''
    Caluclates the sum of the weighted likelihood functions to be maximised by scipy.optimise.minimize.

    Inputs:
        alpha:          Float:          Initial value for alpha.
        w:              np.Array:       Array of exponentially distributed random vairables.
        y_list:         List:           List of y's with i.i.d. noise, where each y has the same shape as u and represents the observed data.
        sigma_l:        Float:          Variance in likelihood (estimate of the noise in the data).
        mesh:           FEniCS mesh:    Spatial mesh used at each timestep in finite elements method.
        tau:            Float:          Rate of growth of hot wall boundary.
        epsilon:        Float:          Diffusion coefficient from advection-diffusion equation.
        num_steps:      Int:            Number of steps in time calculated by the double-glazing solver.
        dt_min:         Float:          Initial timestepping value (variable time-stepping).
        dt_max:         Float:          Limit of dt (variable time-stepping).
        reg:            Float:          Regularisation constant used in variable time-stepping (hyperparameter).
    Outputs:
        -weighted_sum:  Float:          Minus the sum of the weighted log likelihoods for a given alpha for all y's.           
    '''
    
    # Calculate approximate numerical solution to double glazing problem using given alpha.
    u, __ = master_solver(# nx,
                          # ny,
                          mesh,
                          tau,
                          alpha,
                          epsilon,
                          num_steps,
                          dt_min,
                          dt_max,
                          reg
                          )
    
    # Average data in space.
    u = np.mean(u, axis=1)
    
    # Create array of exponentially weighted log likelihoods.
    weighted_array = np.multiply(w, log_likelihood(u,
                                                   y_list,
                                                   sigma_l
                                                   ))
    
    # Sum over weighted_array.
    weighted_sum = np.sum(weighted_array)

    return -weighted_sum

def bootstrapping(iterations,
                  y_list,
                  sigma_l,
                  # alpha_0,
                  nx,
                  ny,
                  tau,
                  epsilon,
                  num_steps,
                  dt_min,
                  dt_max,
                  reg,
                  tol,
                  bracket
                  ):
    '''
    Performs iterations of the weighted likelihood bootstrapping algorithm to generate a sampling distribution for alpha in the double-glazing
    inverse problem.

    Inputs:
        iterations:     Int:        Desired number of samples from WLB algorithm.
        y_list:         List:       List of y's with i.i.d. noise, where each y has the same shape as u and represents the observed data.
        nx:             Int:        Number of horizontal grid points on mesh.
        ny:             Int:        Number of vertical grid points on mesh. 
        tau:            Float:      Rate of growth of hot wall boundary.
        epsilon:        Float:      Diffusion coefficient from advection-diffusion equation.
        num_steps:      Int:        Number of steps in time calculated by the double-glazing solver.
        dt_min:         Float:      Initial timestepping value (variable time-stepping).
        dt_max:         Float:      Limit of dt (variable time-stepping).
        reg:            Float:      Regularisation constant used in variable time-stepping (hyperparameter).
        tol:            Float:      Tolerance of Brent optimiser used.
        bracket:        Tuple:      Bracket used for Brent optimiser.
    Outputs:
        alpha_list:     List:       List of samples for alpha generated by WLB algorithm.
    '''

    # Generate mesh for numerical solver using Stretch_Mesh.py.
    mesh = stretch_mesh(nx=nx, ny=ny)

    # Initialise list of samples.
    alpha_list = []

    # Begin WLB iterations.
    for _ in range(iterations):
        
        # Generate array of exponential random variables.
        w = np.random.exponential(1, len(y_list))

        # Set up list of arguments for use in scipy.optimise.minimize.
        args = (w,
                y_list,
                sigma_l,
                mesh,
                tau,
                epsilon,
                num_steps,
                dt_min,
                dt_max,
                reg
                )      

        # res = minimize(weighted_likelihood_neg,
        #                float(alpha_0),
        #                args,
        #                'BFGS',
        #                tol=tol
        #                )

        # Find maximum of weighted likelihood sum.
        res = minimize_scalar(weighted_likelihood_neg,
                              bracket,
                              args = args,
                              method = 'brent',                             
                              tol = tol
                              )

        # Take sample as the result of optimisation, and add to sample list.
        alpha = res.x
        alpha_list.append(alpha.copy())
    
    return alpha_list

if __name__ == "__main__":

    import pickle 

    def main():

        # Solver variables.
        tau = 1/10                              # Rate of growth of hot wall.
        epsilon = 1/100                         # Diffusion coefficient.
        num_steps = 60                          # Number of steps to run solver for.
        nx = 16                                 # Number of horizontal grid points on mesh.
        ny = 16                                 # Number of vertical grid points on mesh. 

        # Variable time-stepping variables.                                                                                                          
        dt_min = 1e-3                           # Inital value for dt.
        dt_max = 0.1                            # Limit of dt values as t appraches infty.
        reg = 10                                # Regularisation constant.

        # Bootstrapping variables.
        iterations = 100                        # Number of samples to generate.
        sigma_l = 0.2                           # Vairance in likelihood (estimate of noise).
        n = 100                                 # Number of i.d.d. data sets.

        # Optimiser variables.
        # alpha_0 = 1                           # Initial starting value of alpha for optimiser.
        tol = 1e-5                              # Tolerance of optimiser.
        bracket = (-5,5)                        # Bracket used in Brent scalar optimisation.

        # Data augmentation.
        # alpha_star = 4                        # True value for alpha.   
        var_noise = 0.2                         # Variance of noise added to data.

        # Open file containing high resolution data y.
        with open('Master/Data_eps100_num150000_tau10_alpha0/master_data_32_interpolate', 'rb') as file:
            y_star = pickle.load(file)
        
        # Thin high resolution data to correspond with variable time-stepping.
        y_star_coarse = time_organiser(y_star,
                                       dt_min,
                                       dt_max,
                                       tau,
                                       reg,
                                       num_steps,
                                       tol
                                       )[:-1] # Make removing last element more elegant!


        # mesh = stretch_mesh(nx=nx, ny=ny)

        # y_star_coarse, __ = master_solver(# nx,
        #                                   # ny,
        #                                   mesh,
        #                                   tau,
        #                                   alpha_star,
        #                                   epsilon,
        #                                   num_steps,
        #                                   dt_min,
        #                                   dt_max,
        #                                   reg
        #                                   )

        # Generate n many i.i.d. data sets by adding random noise to y.
        y_list = []
        for _ in range(n):
            y = y_star_coarse + np.random.normal(0, var_noise, np.shape(y_star_coarse))
            y_list.append(y.copy())

        #+ np.random.normal(np.zeros(np.shape(y_star)), var_noise)

        # Run WLB algorithm.
        alpha_list = bootstrapping(iterations,
                                   y_list,
                                   sigma_l,
                                   # alpha_0,
                                   nx,
                                   ny,
                                   tau,
                                   epsilon,
                                   num_steps,
                                   dt_min,
                                   dt_max,
                                   reg,
                                   tol,
                                   bracket
                                   )

        return alpha_list
    
    alpha_list = main()

    print(alpha_list)