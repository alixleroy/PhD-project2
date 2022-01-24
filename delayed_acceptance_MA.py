"""Double-glazing problem with Dirichlet conditions.
  du/dt - epsilon*Laplace(u) + \alpha*dot(grad(u), w) = 0 in the square (-1, 1)*(-1, 1)
  and time domain (0, T]
  u(-1, y) = 0, u(1, y) = 1, u(x, -1) = 0, u(x, 1) = 0

"""

from logging import WARNING
from fenics import *
import numpy as np
from vedo.dolfin import plot, Latex, clear, histogram
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import logging

# To silent unwanted output
logging.getLogger('FFC').setLevel(logging.WARNING)
warnings.filterwarnings('ignore')
set_log_active(False)

def modified_time_double_glazing(parameter,
                                epsilon = 1/200,
                                w = Expression(('2*x[1]*(1-x[0]*x[0])', '-2*x[0]*(1-x[1]*x[1])'), degree=3),
                                num_steps = 10,
                                T = 1.0,
                                nx = 10,
                                ny = 10,
                                k = 1,
                                plot_u=False,
                                plot_u_t=False):
  '''
    parameter: a parameter that controls the intensity of the wind
    epsilon: the viscous parameter
    w: the wind parameter, a vector function or constant
    nx, ny: the number of triangles
    k: order of approximating polynomial
    T: Final time step
    num_steps: number of steps
  '''
  set_log_active(False)

  dt = T / num_steps # time step size

  # Create mesh and define function space
  mesh = RectangleMesh(Point(-1, -1), Point(1, 1), nx, ny)
  V = FunctionSpace(mesh, "P", k)

  # Define boundary condition
  tol = 1E-14
  u_D = Expression('near(x[0], 1, tol) ? 1:0', degree=1, tol=tol)

  def boundary(x, on_boundary):
        return on_boundary

  bc = DirichletBC(V, u_D, boundary)

  # Define initial value
  u_n = project(u_D, V)

  # Define variational problem
  u = TrialFunction(V)
  v = TestFunction(V)
  f = Constant(0.0)

  F = u*v*dx + epsilon*dt*dot(grad(u), grad(v))*dx + dt*parameter*dot(w, grad(u))*v*dx - (u_n + dt*f)*v*dx
  a, L = lhs(F), rhs(F)

  # Time-stepping
  u = Function(V)
  t = 0

  # list to store t and u
  t_list = []
  u_list = []

  for n in range(num_steps):

      # Update current time
      t += dt

      # Compute solution
      solve(a == L, u, bc)
     
      # Plot solution
      if plot_u==True:
        plot(u, cmap='jet', scalarbar='h', text=__doc__)

      # Compute u at the vertices and add them to the list
      u_approx = u.compute_vertex_values(mesh)
      t_list.append(t)
      u_list.append(np.linalg.norm(u_approx))


      # Update previous solution
      u_n.assign(u)
  
  if plot_u_t==True:
      # Plot the norm of u against t
      fig, ax = plt.subplots(figsize=(10,10))

      ax.plot(t_list, u_list)
      ax.set_xlabel('$t$', fontsize=12)
      ax.set_ylabel('$||u||_2$', fontsize=12)
      ax.set_title('$||u||_2$ against time $t$', fontsize=14)

      plt.show()

  u_list = np.array(u_list) 

  return u_list 

def generate_noise_data(sigma, parameter):
  '''
  A function that generates noise data
  '''
  eta = np.random.normal(0, sigma)
  u = modified_time_double_glazing(parameter = parameter)

  y = u + eta

  return y

def prior(w):
    un_p = np.exp(-0.5 * np.matmul(w.T, w)) 

    return un_p

def target(w):
    d = len(w)
    p = (1 / (np.sqrt(2*np.pi))**d ) * np.exp(-0.5 * np.matmul(w.T, w)) 

    return p

def likelihood(A, y, sigma):
    d = len(y)
    l = (1 / (np.sqrt(2*np.pi))**d ) * np.exp((-0.5/sigma**2) * np.linalg.norm(y-A)**2) 

    return l

def proposal(X_prime, X_i, sigma):
    sigma_inv = np.linalg.inv(sigma)   # calculates the inverse of the covariance matrix
    det_sigma = np.linalg.det(sigma)   # calculates the determinant of the covariance matrix
    d = len(X_prime)   # get the length of the random vector
    
    # apply the formula for normal density
    q_normal = np.exp(-0.5 * np.matmul(np.matmul((X_prime-X_i).T, sigma_inv), (X_prime-X_i))) / ((np.sqrt(2*np.pi))**d * det_sigma)
    
    return q_normal


def delayed_acceptance(true_parameter, alpha_0, sigma, dim, num_itr, burning_period):
  '''
  This function implements the delayed-acceptance MCMC using the concept in the paper 'Markov Chain Monte Carlo Using an
  Approximation' by J. Andres Christen and Colin Fox. It works for multi-parameter setting and the prior, likelihood and posterior
  distributions can be modified as desired.
  
  true_parameter: the true parameter used to generate noise data
  alpha_0 = initial value of the estimated parameter
  sigma: the standard deviation
  dim: dimension of the parameter
  num_itr: number of iterations
  burning_period: the percentage of the markov chain needed
  '''
  
   tol = 1e-14  # threshold to compare the current proposal and previous proposal

    # generate data
    y = generate_noise_data(0.05, parameter=true_parameter)

    alpha = [alpha_0]   # initialize the markov chain
    A_0 = modified_time_double_glazing(parameter = alpha_0[0])

    I = np.eye(dim, dim)   # an identity matrix to use with the densities
    
    i = 0

    while num_itr > i:

        alpha_prime = alpha_0 + sigma*np.random.normal(0, 1, dim)   # random walk proposal
        A = modified_time_double_glazing(parameter = alpha_prime[0])

        # calculates the accetance ratio for the first rejection criterion
        acceptance_ratio = (likelihood(A, y, sigma) * prior(alpha_prime)) / (likelihood(A_0, y, sigma) * prior(alpha_0))
        
        # calculates the acceptance probability
        g = min(1, acceptance_ratio)
    
        # acceptance decision
        if g < np.random.uniform(0, 1):
            alpha_prime = alpha_0

        # calculates the accetance ratio for the second rejection criterion
        acceptance_ratio = (target(alpha_prime) * proposal(alpha_0, alpha_prime, sigma**2*I)) / (target(alpha_0) * proposal(alpha_prime, alpha_0, sigma**2*I)) 
        rho = min(1, acceptance_ratio)  

        if abs(alpha_prime - alpha_0) > tol:   # will only proceed to the next stage if the current proposal is different from the previous one
            # acceptance decision
            if rho > np.random.uniform(0, 1):
                alpha.append(alpha_prime)
            else:
                alpha.append(alpha_0)

        A_0 = A
        alpha_0 = alpha_prime
        i += 1

    # To generate a 1-D array for one dimensional parameter
    if dim==1:
        alpha = np.ravel(alpha)
    
    # Cut off the burning part
    alpha = alpha[-int(burning_period*len(alpha)):]
    
    # Create a trace and distribution plots of the estimated parameter
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].plot(range(len(alpha)), alpha)
    ax[0].set_xlabel('$k$')
    ax[0].set_ylabel(r'$ \alpha $')
    ax[0].set_title(r'Trace plot of $\alpha$ after the burning period')

    sns.histplot(alpha, kde=True, ax=ax[1])
    ax[1].set_xlabel(r'$ \alpha $')
    ax[1].set_title(r'Distribution plot of $ \alpha $')

    plt.show()

    return alpha

# call the function
alpha = delayed_acceptance(true_parameter=1.0, alpha_0=np.array([3.0]), sigma=0.1, dim=1, num_itr=2000, burning_period=0.9)
print(alpha)

# call the function
alpha = delayed_acceptance(true_parameter=1.0,
                           alpha_0=np.array([2.0]),
                           sigma=0.1,
                           dim=1,
                           num_itr=1000,
                           burning_period=0.9)
print(alpha)

        
