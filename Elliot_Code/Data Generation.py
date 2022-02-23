import fenics
import numpy as np
import matplotlib.pyplot as plt
import typing
from typing import Callable, Tuple, List, Union

from Solver import double_glazing_setup, double_glazing_time


def log_ratio(y: np.array, u1: np.array, u2:np.array, alpha1: float, alpha2: float, sigma_p: float, sigma_l: float, mu_p: float) -> np.array:
    '''
    
    '''
    return 0.5 * ((((alpha1 - mu_p) ** 2 - (alpha2 - mu_p) ** 2) / sigma_p ** 2) + (np.linalg.norm(y - u1) ** 2 - np.linalg.norm(y - u2) ** 2) / sigma_l ** 2)


def random_walk_metropolis(y: np.array, alpha0: float, iterations: int, sigma_q: float, sigma_p: float, sigma_l: float, mu_p: float, tau: float, epsilon: float, num_steps: int, dt_max: float, nx: int, ny: int, k: int) -> np.array:
    '''
    hello
    '''
    
    alpha_list = [alpha0]

    prop_list = []

    A_list = []

    K_list = []

    alpha1 = alpha0

    setup_list = double_glazing_setup(nx, ny, k, tau)

    u1 = double_glazing_time(setup_list, dt_max, alpha1, epsilon, num_steps)

    for i in range(iterations):


        alpha2 = np.random.normal(alpha1, sigma_q)

        u2 = double_glazing_time(setup_list, dt_max, alpha2, epsilon, num_steps)

        A = log_ratio(y, u1, u2, alpha1, alpha2, sigma_p, sigma_l, mu_p)

        K = np.log(np.random.uniform(0,1))

        if A >= 0 or K <= A:    

            alpha1 = alpha2

        alpha_list.append(alpha1)

        u1 = u2

        prop_list.append(alpha2)

        A_list.append(A)

        K_list.append(K)

        print(i)

    return alpha_list, prop_list, A_list, K_list




if __name__ == "__main__":

    def main():

        alpha_star = 0
        alpha0 = 1
        iterations = 50

        tau = 1/10
        epsilon = 1/200
        num_steps = 100
        dt_max = 0.1
        nx = 30
        ny = 30
        k = 1

        sigma_q = 0.5
        sigma_p = 1
        mu_p = 0
        sigma_l = 0.2
        
        var_noise = 1

        setup_list = double_glazing_setup(nx, ny, k, tau)

        y_star = double_glazing_time(setup_list, dt_max, alpha_star, epsilon, num_steps)

        y = y_star + np.random.normal(np.zeros(np.shape(y_star)), var_noise)

        return random_walk_metropolis(y, alpha0, iterations, sigma_q, sigma_p, sigma_l, mu_p, tau, epsilon, num_steps, dt_max, nx, ny, k)

    alpha_list, prop_list, A_list, K_list = main()
    print(alpha_list)
    print(prop_list)
    print(A_list)
    print(K_list)

    plt.hist(alpha_list, 100, (-5, 5))
    plt.show()
