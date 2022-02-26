import numpy as np
import matplotlib.pyplot as plt
from fenics import *
from vedo.dolfin import plot
import warnings
import logging

# To silent unwanted output
logging.getLogger('FFC').setLevel(logging.WARNING)
warnings.filterwarnings('ignore')
set_log_active(False)


def exponential_time_double_glazing(parameter,
                                    tau,
                                    epsilon,
                                    w,
                                    dt,
                                    T,
                                    nx,
                                    ny,
                                    plot_u=False,
                                    plot_u_t=False):
  '''
    parmeter: a constant that controls the strength of the wind 
    tau: a parameter that controls how heated the wall is
    epsilon: the viscous parameter
    w: the wind parameter, a vector function or constant
    nx, ny: the number of triangles
    k: order of approximating polynomial
    T: Final time step
    num_steps: number of steps
  '''

  vtkfile = File('Alixcode/presentation/solution_try.pvd')


  set_log_active(False)

  # Create mesh and define function space
  mesh = RectangleMesh(Point(-1, -1), Point(1, 1), nx, ny)
  V = FunctionSpace(mesh, "P", 1)

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

  F = u*v*dx + epsilon*dt*dot(grad(u), grad(v))*dx + dt*exp(parameter)*dot(w, grad(u))*v*dx - (u_n + dt*f)*v*dx
  a, L = lhs(F), rhs(F)

  # Time-stepping
  u = Function(V)
  t = 0

  # list to store t and u
  t_list = []
  u_list = []

  while t<T:

      # Update current time
      t += dt
      t=round(t,6)
      u_D.t = t
      t_list.append(t)

      # Compute solution
      solve(a == L, u, bc)
     
    #   # Plot solution
    #   if plot_u==True:
    #     plot(u, cmap='jet', scalarbar='h', text=__doc__)

      # Compute u at the vertices and add them to the list
      u_approx = u.compute_vertex_values(mesh)

      u_list.append(np.linalg.norm(u_approx))
      #u_list.append(u_approx)


      # Update previous solution
      u_n.assign(u)

      # Save the file 
      vtkfile << (u, t)
  
  if plot_u_t==True:
      # Plot the norm of u against t
      fig, ax = plt.subplots(figsize=(10,10))

      ax.plot(t_list, u_list)
      ax.set_xlabel('$t$', fontsize=12)
      ax.set_ylabel('$||u||_2$', fontsize=12)
      ax.set_title('$||u||_2$ against time $t$', fontsize=14)

      plt.show()

  u_list = np.array(u_list) 

  return u_list,t_list



# exponential_time_double_glazing(parameter=1.0,
#                                 tau = 1/10 ,
#                                 epsilon = 1/200,
#                                 w = Expression(('2*x[1]*(1-x[0]*x[0])', '-2*x[0]*(1-x[1]*x[1])'), degree=3),
#                                 dt = 0.1,
#                                 T = 5,
#                                 nx = 20,
#                                 ny = 20,
#                                 plot_u=True,
#                                 plot_u_t=True)




# def fenics_essentials_exponential(nx, ny, k, f_val):

#   # Create mesh and define function space
#   mesh = RectangleMesh(Point(-1, -1), Point(1, 1), nx, ny)
#   V = FunctionSpace(mesh, "P", k)

#   # Define variational problem
#   u = TrialFunction(V)
#   v = TestFunction(V)
#   f = Constant(f_val)

#   return mesh, V, u, v, f


# def pred_exponential_time_dg(parameter,
#                             tau,
#                             mesh,
#                             V,
#                             u,
#                             v,
#                             f,
#                             epsilon = 1/200,
#                             w = Expression(('2*x[1]*(1-x[0]*x[0])', '-2*x[0]*(1-x[1]*x[1])'), degree=3),
#                             num_steps = 50,
#                             T = 5.0,
#                             plot_u=False,
#                             plot_u_t=False):
  
#   set_log_active(False)

#   dt = T / num_steps # time step size

#   # Define boundary condition
#   tol = 1E-14
#   u_D = Expression('near(x[0], 1, tol) ? (1-pow(x[1], 4))*(1-exp(-t / tau)):0', degree=4, tol=tol, tau=tau, t=0)

#   def boundary(x, on_boundary):
#         return on_boundary

#   bc = DirichletBC(V, u_D, boundary)

#   # Define initial value
#   u_n = project(u_D, V)

#   # Define the linear and bilinear forms
#   F = u*v*dx + epsilon*dt*dot(grad(u), grad(v))*dx + dt*parameter*dot(w, grad(u))*v*dx - (u_n + dt*f)*v*dx
#   a, L = lhs(F), rhs(F)

#   # Time-stepping
#   u = Function(V)
#   t = 0

#   # list to store t and u
#   t_list = []
#   u_list = []

#   for n in range(num_steps):

#       # Update current time
#       t += dt
#       u_D.t = t

#       # Compute solution
#       solve(a == L, u, bc)
     
#       # Plot solution
#       if plot_u==True:
#         plot(u, cmap='jet', scalarbar='h', text=__doc__)

#       # Compute u at the vertices and add them to the list
#       u_approx = u.compute_vertex_values(mesh)
#       t_list.append(t)
#       #u_list.append(np.linalg.norm(u_approx))
#       u_list.append(u_approx)


#       # Update previous solution
#       u_n.assign(u)
  
#   if plot_u_t==True:
#       # Plot the norm of u against t
#       fig, ax = plt.subplots(figsize=(10,10))

#       ax.plot(t_list, u_list)
#       ax.set_xlabel('$t$', fontsize=12)
#       ax.set_ylabel('$||u||_2$', fontsize=12)
#       ax.set_title('$||u||_2$ against time $t$', fontsize=14)

#       plt.show()

#   u_list = np.array(u_list) 

#   return u_list 

