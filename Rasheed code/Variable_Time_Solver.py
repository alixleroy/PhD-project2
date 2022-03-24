import numpy as np
import matplotlib.pyplot as plt
from fenics import *
from vedo.dolfin import plot
import warnings
import logging
import matplotlib.pyplot as plt

# To silent unwanted output
logging.getLogger('FFC').setLevel(logging.WARNING)
warnings.filterwarnings('ignore')
set_log_active(False)


def var_exponential_time_double_glazing(parameter,
                                        t_list,
                                        tau = 1/10,
                                        epsilon = 1/100,
                                        w = Expression(('2*x[1]*(1-x[0]*x[0])', '-2*x[0]*(1-x[1]*x[1])'), degree=3),
                                        nx = 10,
                                        ny = 10,
                                        k = 1,
                                        plot_u=False,
                                        plot_u_t=False):

  '''

  A function that solves the double glazing problem for the case of variable time steps

  Arguments:

    parmeter:   a constant that controls the strength of the wind 
    t_list:     a list of time steps 
    tau:        a parameter that controls how heated the wall is
    epsilon:    the viscous parameter
    w:          the wind parameter, a vector function or constant
    nx, ny:     the number of triangles
    k:          order of approximating polynomial 
    plot_u:     a boolean, either to plot u or not
    plot_u_t:   a boolean, either to plot the average temperature as a function of time or not

  '''
  # Silent FEniCS warnings
  set_log_active(False)
  
  # Initialize the time step
  dt = t_list[0]

  # Create mesh and define function space
  mesh = RectangleMesh(Point(-1, -1), Point(1, 1), nx, ny)
  V = FunctionSpace(mesh, "P", k)

  # Define boundary condition
  tol = 1E-14
  u_D = Expression('near(x[0], 1, tol) ? (1-pow(x[1], 4))*(1-exp(-t / tau)):0', degree=4, tol=tol, tau=tau, t=0)

  def boundary(x, on_boundary):
        return on_boundary

  bc = DirichletBC(V, u_D, boundary)

  # Define initial value
  u_n = project(u_D, V)

  # Define variational problem
  u = TrialFunction(V)
  v = TestFunction(V)
  f = Constant(0.0)
  dt = Constant(dt)

  F = u*v*dx + epsilon*dt*dot(grad(u), grad(v))*dx + dt*exp(parameter)*dot(w, grad(u))*v*dx - (u_n + dt*f)*v*dx
  a, L = lhs(F), rhs(F)

  # define the solution function
  u = Function(V)

  # list to store t and u
  u_list = []

  for t in t_list:

      # Update the boundary condition
      u_D.t = t

      # update dt 
      dt.assign(t)

      # Compute solution
      solve(a == L, u, bc)

      # Compute u at the vertices and add them to the list
      u_approx = u.compute_vertex_values(mesh)
      u_list.append(np.linalg.norm(u_approx))
      #u_list.append(u_approx)
      
      # Plot solution
      if plot_u==True:
        plot(u, cmap='jet', scalarbar='h', text=__doc__)

      # Update previous solution
      u_n.assign(u)
  
  if plot_u_t==True:
      # Plot the norm of u against t
      fig, ax = plt.subplots(figsize=(10,10))

      ax.plot(t_list, u_list)
      ax.set_xlabel('$t$')
      ax.set_ylabel('$||u||_2$')
      ax.set_title('$||u||_2$ against time $t$')

      plt.show()

  u_list = np.array(u_list) 

  return u_list


def var_pred_exponential_time_dg(parameter,
                                t_list,
                                mesh,
                                V,
                                u,
                                v,
                                tau=1/10,
                                f=0.0,
                                epsilon = 1/200,
                                w = Expression(('2*x[1]*(1-x[0]*x[0])', '-2*x[0]*(1-x[1]*x[1])'), degree=3),
                                plot_u=False,
                                plot_u_t=False):
  
  '''
  
  A function that solves the double glazing problem for the case of variable time steps

  Arguments:

    parmeter:   a constant that controls the strength of the wind 
    t_list:     a list of time steps 
    mesh:       the solution mesh
    V:          the solution space
    u:          trial function
    v:          test function
    f:          the source term
    tau:        a parameter that controls how heated the wall is
    epsilon:    the viscous parameter
    w:          the wind parameter, a vector function or constant
    plot_u:     a boolean, either to plot u or not
    plot_u_t:   a boolean, either to plot the average temperature as a function of time or not

    '''
  # Silent FEniCS warning message
  set_log_active(False)
  
  # Initialize the time step
  dt = t_list[0]

  # Define boundary condition
  tol = 1E-14
  u_D = Expression('near(x[0], 1, tol) ? (1-pow(x[1], 4))*(1-exp(-t / tau)):0', degree=4, tol=tol, tau=tau, t=0)

  def boundary(x, on_boundary):
        return on_boundary

  bc = DirichletBC(V, u_D, boundary)

  # Define initial value
  u_n = project(u_D, V)

  # Define variational problem
  u = TrialFunction(V)
  v = TestFunction(V)
  f = Constant(0.0)
  dt = Constant(dt)

  F = u*v*dx + epsilon*dt*dot(grad(u), grad(v))*dx + dt*exp(parameter)*dot(w, grad(u))*v*dx - (u_n + dt*f)*v*dx
  a, L = lhs(F), rhs(F)

  # define the solution function
  u = Function(V)

  # list to store t and u
  u_list = []

  for t in t_list:

      # Update the boundary condition
      u_D.t = t

      # update dt 
      dt.assign(t)

      # Compute solution
      solve(a == L, u, bc)

      # Compute u at the vertices and add them to the list
      u_approx = u.compute_vertex_values(mesh)
      u_list.append(np.linalg.norm(u_approx))
      #u_list.append(u_approx)
      
      # Plot solution
      if plot_u==True:
        plot(u, cmap='jet', scalarbar='h', text=__doc__)

      # Update previous solution
      u_n.assign(u)
  
  if plot_u_t==True:
      # Plot the norm of u against t
      fig, ax = plt.subplots(figsize=(10,10))

      ax.plot(t_list, u_list)
      ax.set_xlabel('$t$')
      ax.set_ylabel('$||u||_2$')
      ax.set_title('$||u||_2$ against time $t$')

      plt.show()

  u_list = np.array(u_list) 

  return u_list