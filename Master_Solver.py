'''
Master solver used in the likelihood function for all sampling methods.
'''

import numpy as np
from fenics import *
#import warnings

from Time_Stepping import next_dt

# To silent unwanted output
# logging.getLogger('FFC').setLevel(logging.WARNING)
# warnings.filterwarnings('ignore')
# set_log_active(False)

def master_solver(#nx,
                  #ny,
                  mesh,
                  tau,
                  alpha,
                  epsilon,
                  num_steps,
                  dt_min,
                  dt_max,
                  reg
                  ):
  '''
  Uses FEniCS Project package to numerically solve the double-glazing problem using the finite elements method.

  Inputs:
      mesh:         FEniCS mesh:    Spatial mesh used at each timestep in finite elements method.# RWMH parameters.
        iterations = 100                        # Number of samples to generate.
        sigma_q = 0.1                           # Vairance of normal proposal distribution.
        sigma_p = 1                             # Variance of normal prior.
        mu_p = 0                                # Mean of normal prior.
        sigma_l = 0.2                           # Variance in likelihood (estimate of noise in data):
      tau:          Float:          Rate of growth of hot wall boundary.      
      alpha:        Float:          Parameter for the wind speed.
      epsilon:      Float:          Diffusion coefficient from advection-diffusion equation.        
      num_steps:    Int:            Number of steps in time to calculate.
      dt_min:       Float:          Initial timestepping value (variable time-stepping).
      dt_max:       Float:          Limit of dt (variable time-stepping).
      reg:          Float:          Regularisation constant used in variable time-stepping (hyperparameter).
  Outputs:
      u_list:       List:           List representing point values of the numerical solution at each vertice in time.
      t_list:       List:           List of actual times at which the solution is evaulated.
  '''
  # Define function space.
  V = FunctionSpace(mesh, "P", 1)

  # Define boundary condition.
  tol = 1E-14
  u_D = Expression('near(x[0], 1, tol) ? (1-pow(x[1], 4))*(1-exp(-t / tau)):0', degree=4, tol=tol, tau=tau, t=0)

  def boundary(x, on_boundary):
        return on_boundary

  bc = DirichletBC(V, u_D, boundary)

  # Define initial value.
  u_n = project(u_D, V)

  # Define variational problem.
  u = TrialFunction(V)
  v = TestFunction(V)
  f = Constant(0.0)

  # Initialise the time step.
  dt = dt_min
  dt_f = Constant(dt)

  # Define the wind.
  w = Expression(('2*x[1]*(1-x[0]*x[0])', '-2*x[0]*(1-x[1]*x[1])'), alpha = alpha, degree=3)

  F = u * v * dx + epsilon * dt_f * dot(grad(u), grad(v)) * dx + dt_f * exp(alpha) * dot(w, grad(u)) * v * dx - (u_n + dt_f * f) * v * dx
  a, L = lhs(F), rhs(F)

  # Time-stepping
  u = Function(V)
  t = 0

  # Lists to store t and u.
  u_list = []
  t_list = []

  for _ in range(num_steps):

      # Compute solution.
      solve(a == L, u, bc)

      # Compute u at the vertices and add them to the list.
      u_approx = u.compute_vertex_values(mesh)
      u_list.append(u_approx.copy())

      # Update previous solution.
      u_n.assign(u)
  
      # Time.
      t_list.append(t)

      # Update current time
      dt = next_dt(dt_min, dt_max, t, tau, reg)
      t += dt
      t = round(t,5) # t%dt_res should be zero 
      dt_f.assign(dt)

      u_D.t = t #update the time in the boundary condition

  return u_list, t_list

if __name__ == '__main__':

  import matplotlib.pyplot as plt

  from Stretch_Mesh import stretch_mesh

  def main():

    # Coarse.
    alpha = 0                               # Parameter determining wind speed (exp(alpha)).   
    tau = 1/10                              # Rate of growth of hot wall.
    epsilon = 1/200                         # Diffusion coefficient.
    num_steps = 100                         # Number of steps to run solver for.
    nx = 32                                 # Number of horizontal grid points on mesh.
    ny = 32                                 # Number of vertical grid points on mesh. 

    # Variable time-stepping variables.                                                                                                          
    dt_min = 1e-3                           # Inital value for dt.
    dt_max = 0.1                            # Limit of dt values as t appraches infty.
    reg = 10                                # Regularisation constant.

    # Generate mesh using Stretch_Mesh.py.
    mesh = stretch_mesh(nx=nx, ny=ny)

    # Run solver.
    u_list, t_list = master_solver(#nx,
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

    return u_list, t_list


  u_list, t_list = main()

  print(np.shape(u_list))
  plt.plot(np.linalg.norm(np.array(u_list), axis=1))