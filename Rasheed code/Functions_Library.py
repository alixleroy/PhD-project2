import pickle
import numpy as np
from fenics import *
import matplotlib.pyplot as plt
from scipy.special import gamma
import statsmodels.api as sm
import seaborn as sns

def stretch_mesh(
        BottomPoint = (-1,-1),
        TopPoint = (1,1),
        nx = 32,
        ny = 32,
        s = 2.5
    ):
    '''
    Inputs:
        BottomPoint:    Tuple       Bottom left point of rectangle domain.
        TopPoint:       Tuple       Top right point of rectangle domain.
        nx:             Int         Number of vertices on horizontal axis.
        ny:             Int         Number of vertices on vertical axis.
        s:              Float       Stretching coefficient.
    Outputs:
        mesh:           Mesh        Mesh stretched towards right vertical boundary.
    '''
    # Create uniform rectangle mesh.
    a, b = BottomPoint[0], TopPoint[0]
    mesh = RectangleMesh(Point(a, BottomPoint[1]), Point(b, TopPoint[1]), nx, ny)
    ### Stretch horizontally to right vertical boundary.
    x = mesh.coordinates()[:,0]  
    y = mesh.coordinates()[:,1]
    # Stretching function.
    def denser(x,y):
        return [b + (a-b)*((x-a)/(b-a))**s, y]
    x_bar, y_bar = denser(x, y)
    xy_bar_coor = np.array([x_bar, y_bar]).transpose()
    mesh.coordinates()[:] = xy_bar_coor

    # Create mesh and define function space
    V = FunctionSpace(mesh, "P", 1)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)

    return mesh, V, u, v


def prior(x):
  return np.exp(-0.5 * x**2)/(np.sqrt(2*np.pi))

def likelihood(A, y, sigma):
    y = np.linalg.norm(y, axis=1)
    return (1 / (np.sqrt(2*np.pi))*sigma) * np.exp((-0.5/sigma**2) * np.linalg.norm(y-A)**2)  # the parameter space is just a subset of the real line

def proposal(xp, x, sigma):
    return np.exp(-0.5*(xp-x)**2/sigma**2) / (np.sqrt(2*np.pi)*sigma)

def gamma_proposal(x, k, theta):
  return (x**(k-1) * exp(-x/theta)) / (theta**k * gamma(k))

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

    sns.histplot(alpha_dt, kde=True, stat="probability", ax=ax[0,1])
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

def compare_diagnosis_plots(mcmc_chains,
                            mcmc_acceptance_ratios,
                            lag,
                            nbins,
                            labels,
                            file_name):
    '''
    A function that compares the diagnostic plots of two chains

    Arguments:
        mcmc_chains:                tuple containing the chains
        mcmc_acceptance_ratios:     tuple containing the acceptance ratios
        lag:                        autocorrelation lag
        nbins:                      number of bins for histogram
        labels:                     tup;le containing the figure labels
        file_name:                  name to save generated figure with
    '''
    # get the list of iterations
    k = list(range(len(mcmc_chains[0])))

    # calculate autocorrelations
    acf_1 = sm.tsa.acf(mcmc_chains[0], nlags=lag)
    laglist_1 = list(range(len(acf_1)))

    acf_2 = sm.tsa.acf(mcmc_chains[1], nlags=lag)
    laglist_2 = list(range(len(acf_2)))

    # Create a trace and distribution plots of the estimated parameter
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0,0].plot(k, mcmc_chains[0], label=labels[0])
    ax[0,0].plot(k, mcmc_chains[1], label=labels[1])
    ax[0,0].set_xlabel('$k$')
    ax[0,0].set_ylabel(r'$ \alpha $')
    ax[0,0].set_title(r'Trace plot of $\alpha$')
    ax[0,0].legend()

    sns.histplot(mcmc_chains[0], kde=True, stat="probability", bins=nbins, label=labels[0], ax=ax[0,1])
    sns.histplot(mcmc_chains[1], kde=True, stat="probability", bins=nbins, label=labels[1], ax=ax[0,1])
    ax[0,1].set_xlabel(r'$ \alpha $')
    ax[0,1].set_title(r'Distribution plot of $ \alpha $')
    ax[0,1].legend()
    
    ax[1,0].plot(range(0, len(mcmc_acceptance_ratios[0])), mcmc_acceptance_ratios[0], label=labels[0])
    ax[1,0].plot(range(0, len(mcmc_acceptance_ratios[1])), mcmc_acceptance_ratios[1], label=labels[1])
    ax[1,0].set_xlabel('$k$')
    ax[1,0].set_ylabel(r'Acceptance ratio')
    ax[1,0].set_title(r'Ratio of accepted state to the total number of states')
    ax[1,0].legend()

    ax[1,1].plot(laglist_1, acf_1, label=labels[0])
    ax[1,1].plot(laglist_2, acf_2, label=labels[1])
    ax[1,1].set_xlabel(r'lags $(k)$')
    ax[1,1].set_ylabel(r'$ACF$')
    ax[1,1].set_title(r'Autocorrelation plot')
    ax[1,1].legend()


    plt.savefig(fname=file_name, format='png')

    plt.show()

def load_data(fname):
    '''
    A function that loads and return data

    Argument:
          fname: the name of the file
    '''
    with open(fname, 'rb') as file:
        data = pickle.load(file)

    return data
