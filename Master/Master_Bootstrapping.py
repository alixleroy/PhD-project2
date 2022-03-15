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

    log_likelihood_list = []

    for y in y_list:
        f = np.log(2/(sigma_l * np.sqrt(2 * np.pi)))- (1/(2 * sigma_l ** 2)) * np.linalg.norm(y - A_mat(u))
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
    
    weighted_array = np.multiply(w, log_likelihood(u,
                                                   y_list,
                                                   sigma_l
                                                   ))
    
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
    '''

    mesh = stretch_mesh(nx=nx, ny=ny)

    alpha_list = []

    for _ in range(iterations):
        
        w = np.random.exponential(1, len(y_list))

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

        res = minimize_scalar(weighted_likelihood_neg,
                              bracket,
                              args = args,
                              method = 'brent',                             
                              tol = tol
                              )

        alpha = res.x
        print(alpha)
        alpha_0 = alpha

        alpha_list.append(alpha.copy())
    
    return alpha_list

if __name__ == "__main__":

    import pickle 

    def main():

        # Solver variables.
        tau = 1/10                              # Rate of growth of hot wall.
        epsilon = 1/100                         # Diffusion coefficient.
        num_steps = 60                          # Number of steps to run solver for.
        nx = 32                                 # Number of horizontal grid points on mesh.
        ny = 32                                 # Number of vertical grid points on mesh. 

        # Variable time-stepping variables.                                                                                                          
        dt_min = 1e-3                           # Inital value for dt.
        dt_max = 0.1                            # Limit of dt values as t appraches infty.
        reg = 10                                # Regularisation constant.

        # Bootstrapping variables.
        iterations = 100                        # Number of samples to generate.
        sigma_l = 0.2                           # Vairance in likelihood (estimate of noise).
        n = 100                                 # Number of i.d.d. datapoints.

        # Optimiser variables.
        # alpha_0 = 1                           # Initial starting value of alphafor optimiser.
        tol = 1e-5                              # Tolerance of optimiser.
        bracket = (-5,5)                        # Bracket used in Brent scalar optimisation.

        # Data augmentation.
        # alpha_star = 4                        # True value for alpha.   
        var_noise = 0.2                         # Variance of noise added to data.

        with open('Master/Data_eps100_num150000_tau10_alpha0/master_data_32_interpolate', 'rb') as file:
            y_star = pickle.load(file)
        
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

        y_list = []
        for _ in range(n):
            y = y_star_coarse + np.random.normal(0, var_noise, np.shape(y_star_coarse))
            y_list.append(y.copy())

        #+ np.random.normal(np.zeros(np.shape(y_star)), var_noise)

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