from fenics import *
import numpy as np
from vedo.dolfin import plot, Latex, clear, histogram
import matplotlib.pyplot as plt
import pickle 
import time

from Function_Solver import solver_para, solver_run


def time_organiser(f1_list_raw, f2_list_raw, others):
    '''
    Takes two function lists from the fenics solver, thins and organises them such that their positions correspond to the same times,
    '''
    f1_list = f1_list_raw
    f2_list = f2_list_raw
    
    return f1_list, f2_list


def questionable_norm(f1_list, f2_list, number_of_timesteps_considered = 10, initial_jump = 5) -> float:
    '''
    Calculate the questionable norm error between the two funcitons f1 & f2 defined on [-1,1]^2.
    '''
    L2_list = []
        
    k = len(f1_list)
    sparstiy = int((np.floor(k - initial_jump) / number_of_timesteps_considered))
    
    for i in range(initial_jump, k, sparstiy):
        L2_list.append(errornorm(f1_list[i], f2_list[i], 'L2'))

    return np.linalg.norm(L2_list)  


def log_ratio(y: np.array, u1: np.array, u2:np.array, alpha1: float, alpha2: float, sigma_p: float, sigma_l: float, mu_p: float, f1_quest_norm: float, f2_quest_norm: float) -> np.array:
    '''
    Calculate the log acceptance ratio between two functions using the questionable norm.
    '''
    return 0.5 * ((((alpha1 - mu_p) ** 2 - (alpha2 - mu_p) ** 2) / sigma_p ** 2) + (f1_quest_norm ** 2 - f2_quest_norm ** 2)/ sigma_l ** 2)


def random_walk_metropolis(alpha0, y, sigma_q, sigma_p, sigma_l, mu_p, nx, ny, tau, epsilon, iterations, num_steps, number_of_timesteps_considered):
    '''
    Carries out iterations of the random walk Metropolis-Hastings Algorithm using likelihood estimator.
    '''

    V, u, v, u_n, f, u_D, bc, mesh = solver_para(nx, ny, tau)
    
    alpha_list = [alpha0]

    alpha1 = alpha0

    u1 = solver_run(alpha1, V, u, v, u_n, f, u_D, bc, epsilon, num_steps)

    u1_quest_norm = questionable_norm(y, u1, number_of_timesteps_considered)

    for i in range(iterations):

        alpha2 = np.random.normal(alpha1, sigma_q)
        u2 = solver_run(alpha2, V, u, v, u_n, f, u_D, bc, epsilon, num_steps)
        u2_quest_norm = questionable_norm(y, u2, number_of_timesteps_considered)

        A = log_ratio(y, u1, u2, alpha1, alpha2, sigma_p, sigma_l, mu_p, u1_quest_norm, u2_quest_norm)

        if A >= 0 or np.log(np.random.uniform(0,1)) <= A:
            alpha1 = alpha2

        alpha_list.append(alpha1)
        
        u1 = u2
        u1_quest_norm = u2_quest_norm

        print(i)

    return alpha_list


if __name__ == "__main__":

    def main():

        alpha_star = 0
        alpha0 = 1
        iterations = 10
        tau = 1/10
        epsilon = 1/20
        num_steps = 100
        nx = 36
        ny = 36

        sigma_q = 0.1
        sigma_p = 1
        mu_p = 0
        sigma_l = 0.2

        number_of_timesteps_considered = 10

        var_noise = 0.3

        V, u, v, u_n, f, u_D, bc, mesh = solver_para(nx, ny, tau)
        y_star = solver_run(alpha_star, V, u, v, u_n, f, u_D, bc, epsilon, num_steps)
        # noise = np.random.normal(np.zeros((nx + 1)*(ny + 1)), var_noise)
        # noise_function = Function(V)
        # noise_function.vector()[:] = noise
        
        for f in y_star:
            noise = np.random.normal(np.zeros((nx + 1)*(ny + 1)), var_noise)
            # noise_function = Function(V)
            f.vector()[:] += noise
            # y.append(f + noise_function)

        y = y_star

        #+ np.random.normal(np.zeros(np.shape(y_star)), var_noise)

        alpha_list = random_walk_metropolis(alpha0, y, sigma_q, sigma_p, sigma_l, mu_p, nx, ny, tau, epsilon, iterations, num_steps, number_of_timesteps_considered)

        return alpha_list
    
    alpha_list = main()

    print(alpha_list)


