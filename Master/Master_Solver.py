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
      t = round(t, 10) # avoid numerical errors
      dt_f.assign(dt)

      u_D.t = t #update the time in the boundary condition

  return u_list, t_list

if __name__ == '__main__':

  from vedo.dolfin import plot, Latex, clear, histogram
  import matplotlib.pyplot as plt

  from Stretch_Mesh import stretch_mesh

  def main():

    # Coarse.
    tau = 1/10
    epsilon = 1/200
    num_steps = 100
    nx = 32
    ny = 32
    alpha = 0

    # Time stepping.
    dt_min = 1e-3
    dt_max = 0.1
    reg = 10

    mesh = stretch_mesh(nx=nx, ny=ny)

    return master_solver(#nx,
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

  u_list, t_list = main()

  print(np.shape(u_list))
  plt.plot(np.linalg.norm(np.array(u_list), axis=1))