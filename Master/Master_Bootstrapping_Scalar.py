'''
Implementation of weighted likelihood bootstrapping algorithm for the double-glazing inverse problem. Referece paper: "Introducing prior information in Weighted Likelihood Bootstrap
with applications to model misspecification" by Emily Pompe. Scalar version (might not be i.i.d.)
'''
from doctest import master
import numpy as np
from scipy.optimize import minimize
from Stretch_Mesh import stretch_mesh

from Master_Solver import master_solver
from Time_Stepping import time_organiser

def A_mat(u):
    '''
    Format u.
    '''
    return u

def log_likelihood(u,
                   y,
                   sigma_l
                   ):

    return np.log(2/(sigma_l * np.sqrt(2 * np.pi)))- (1/(2 * sigma_l ** 2)) * np.linalg.norm(y - A_mat(u))

def weighted_likelihood_neg(alpha, w, y, sigma_l, mesh, tau, epsilon, num_steps, dt_min, dt_max, reg):

    u = master_solver(#nx,
                      #ny,
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
                                                   y,
                                                   sigma_l
                                                   ))
    
    weighted_sum = np.sum(weighted_array)

    return -weighted_sum

def bootstrapping(iterations,
                  y,
                  sigma_l,
                  alpha_0,
                  nx,
                  ny,
                  tau,
                  epsilon,
                  num_steps,
                  dt_min,
                  dt_max,
                  reg
                  ):
    '''
    Performs iterations of the weighted likelihood bootstrapping algorithm to generate a sampling distribution for alpha in the double-glazing
    inverse problem.
    '''

    mesh = stretch_mesh(nx=nx, ny=ny)

    alpha_list = []

    for _ in range(iterations):
        
        w = np.random.exponential(1,  np.shape(y))
        args = (w, y, sigma_l, mesh, tau, epsilon, num_steps, dt_min, dt_max, reg)
        alpha = minimize(weighted_likelihood_neg, alpha_0, args)
        alpha = alpha.x[0]
        print(alpha)
        alpha_0 = alpha

        alpha_list.append(alpha)
    
    return alpha_list

if __name__ == "__main__":

    import pickle 

    def main():

        # Solver variables.
        tau = 1/10
        epsilon = 1/100
        num_steps = 60
        nx = 32
        ny = 32
        dt_min = 1e-3
        dt_max = 0.1
        reg = 10

        # Bootstrapping variables.
        iterations = 100
        sigma_l = 0.2

        # Optimiser variables.
        alpha_0 = 1

        # Data augmentation.
        alpha_star = 4
        var_noise = 0.3


        with open('Master/Data_eps100_num150000_tau10_alpha0/master_data_32_interpolate', 'rb') as file:
            y_star = pickle.load(file)
        
        y_star_coarse = time_organiser(y_star,
                                       dt_min,
                                       dt_max,
                                       tau,
                                       reg,
                                       num_steps
                                       )[:-1] # Make removing last element more elegant!


        # mesh = stretch_mesh(nx=nx, ny=ny)

        # y_star_coarse, __ = master_solver(#nx,
        #                                   #ny,
        #                                   mesh,
        #                                   tau,
        #                                   alpha_star,
        #                                   epsilon,
        #                                   num_steps,
        #                                   dt_min,
        #                                   dt_max,
        #                                   reg
        #                                   )

        y = y_star_coarse + np.random.normal(0, var_noise, np.shape(y_star_coarse))

        #+ np.random.normal(np.zeros(np.shape(y_star)), var_noise)

        alpha_list = bootstrapping(iterations,
                                   y,
                                   sigma_l,
                                   alpha_0,
                                   nx,
                                   ny,
                                   tau,
                                   epsilon,
                                   num_steps,
                                   dt_min,
                                   dt_max,
                                   reg
                                   )

        return alpha_list
    
    alpha_list = main()

    print(alpha_list)