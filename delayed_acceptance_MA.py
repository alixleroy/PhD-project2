import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from exponential_time_double_glazing import *
from fenics import *
import warnings
import logging
import time
import statsmodels.api as sm

# To silent unwanted output
logging.getLogger('FFC').setLevel(logging.WARNING)
warnings.filterwarnings('ignore')
set_log_active(False)

def prior(w):
    #return np.exp(-0.5 * w**2)
    return np.exp(-w)

def target(w):
    return (1 / (np.sqrt(2*np.pi))) * np.exp(-0.5 * w**2)

def likelihood(A, y, sigma):
    return (1 / (np.sqrt(2*np.pi))) * np.exp((-0.5/sigma**2) * np.linalg.norm(y-A)**2)  # the parameter space is just a subset of the real line

def proposal(X_prime, X_i, sigma):
    return np.exp(-0.5*(X_prime-X_i)**2/sigma**2) / (np.sqrt(2*np.pi)*sigma)

def generate_noise_data(sigma, parameter, tau):
  '''
  A function that generates noise data
  '''
  u = exponential_time_double_glazing(parameter=parameter, tau=tau, nx=10, ny=10, num_steps=100, T=5.0)
  eta = np.random.normal(0, sigma, (u.shape))
  y = u + eta

  return y

def mcmc_diagnosis_plots(mcmc_chain, mcmc_acceptance_ratio, burn_in_period, lag, file_name):

    # Cut off the burning part
    l = len(mcmc_chain)
    burn = int(burn_in_period*l)
    ind_ = list(range(l))
    burn_ind, alpha_ind = ind_[:burn+1], ind_[burn:]
    burn_dt, alpha_dt = mcmc_chain[:burn+1], mcmc_chain[burn:]

    # calculate autocorrelations
    acf = sm.tsa.acf(mcmc_chain, nlags=lag)
    laglist = list(range(len(acf)))

    # Create a trace and distribution plots of the estimated parameter
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0,0].plot(burn_ind, burn_dt, 'r-', label=r'Burn-in period')
    ax[0,0].plot(alpha_ind, alpha_dt, 'b-', label=r'After burn-in')
    ax[0,0].set_xlabel('$k$')
    ax[0,0].set_ylabel(r'$ \alpha $')
    ax[0,0].set_title(r'Trace plot of $\alpha$ before and after the burn-in period')
    ax[0,0].legend()

    sns.histplot(alpha_dt, kde=True, ax=ax[0,1])
    ax[0,1].set_xlabel(r'$ \alpha $')
    ax[0,1].set_title(r'Distribution plot of $ \alpha $ after the burn-in period')
    
    ax[1,0].plot(range(0, len(mcmc_acceptance_ratio)), mcmc_acceptance_ratio)
    ax[1,0].set_xlabel('$k$')
    ax[1,0].set_ylabel(r'$|X_{accept}|/|X_{total}|$')
    ax[1,0].set_title(r'Ratio of accepted state to the total number of states')

    ax[1,1].plot(laglist, acf)
    ax[1,1].set_xlabel(r'lags $(k)$')
    ax[1,1].set_ylabel(r'$ACF$')
    ax[1,1].set_title(r'Autocorrelation plot')

    plt.savefig(fname=file_name, format='png')

    plt.show()


def delayed_acceptance(true_parameter, alpha_0, sigma, num_itr):

    tol = 1e-14  # threshold to compare the current proposal and previous proposal
    tau = 1/100

    # initialize a list to monitor acceptance ratio
    accepted_state_list = []
    acceptance_ratio_list = []

    # generate data
    y = generate_noise_data(sigma=0.02, parameter=true_parameter, tau=tau)
    
    start_time = time.time()   

    alpha = [alpha_0]   # initialize the markov chain

    # initialize a list to monitor acceptance ratio
    accepted_state_list = [alpha_0]
    acceptance_ratio_list = []

    
    # initialize the mesh, function space and source term
    mesh, V, u, v, f = fenics_essentials_exponential(nx=10, ny=10, k=1, f_val=0.0)

    A_0 = pred_exponential_time_dg(alpha_0, tau, mesh, V, u, v, f, epsilon=1/200, num_steps=100, T=5.0)
    #A_0 = exponential_time_double_glazing(parameter=alpha_0, tau=tau)
    i = 0
    
    while num_itr > i:

        alpha_prime = alpha_0 + sigma*np.random.normal(0, 1)   # random walk proposal

        A = pred_exponential_time_dg(alpha_prime, tau, mesh, V, u, v, f, epsilon=1/200, num_steps=100, T=5.0)
        #A = exponential_time_double_glazing(parameter=alpha_prime, tau=tau)

        # calculates the accetance ratio for the first rejection criterion
        acceptance_ratio = (likelihood(A, y, sigma) * prior(alpha_prime)) / (likelihood(A_0, y, sigma) * prior(alpha_0))
        
        # calculates the acceptance probability
        g = min(1, acceptance_ratio)
    
        # acceptance decision
        if g < np.random.uniform(0, 1):
            alpha_prime = alpha_0

        # calculates the accetance ratio for the second rejection criterion
        acceptance_ratio = (target(alpha_prime) * proposal(alpha_0, alpha_prime, sigma)) / (target(alpha_0) * proposal(alpha_prime, alpha_0, sigma)) 
        rho = min(1, acceptance_ratio)  

        if abs(alpha_prime - alpha_0) > tol:   # will only proceed to the next stage if the current proposal is different from the previous one
            # acceptance decision
            if rho > np.random.uniform(0, 1):
                alpha.append(alpha_prime)
                accepted_state_list.append(alpha_prime)
            else:
                alpha.append(alpha_0)
        
        # update the acceptance ratio list 
        acceptance_ratio_list.append(len(accepted_state_list)/len(alpha))

        A_0 = A
        alpha_0 = alpha_prime
        i += 1

    end_time = time.time()
    print('Computation time: ', round(end_time - start_time, 2))
    
    # save the chain and the accepted_ratio list 
    np.save('delayed_MCMC.npy', alpha)
    np.save('delayed_MCMC_accept_ratio.npy', acceptance_ratio_list)

    return alpha
