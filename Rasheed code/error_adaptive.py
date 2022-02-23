
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
                                    epsilon = 1/200,
                                    w = Expression(('2*x[1]*(1-x[0]*x[0])', '-2*x[0]*(1-x[1]*x[1])'), degree=3),
                                    # num_steps = 50,
                                    T = 5.0,
                                    nx = 10,
                                    ny = 10,
                                    dt_tol = 1e-3,
                                    dtold = 0.0001,
                                    k = 1,
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
  set_log_active(False)

#   dt = 1 / num_steps # time step size

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

  # Adaptive time step 
  dt = dtold
  idt = Constant(dt)

  F = u*v*dx + epsilon*idt*dot(grad(u), grad(v))*dx + idt*parameter*dot(w, grad(u))*v*dx - (u_n + idt*f)*v*dx
  a, L = lhs(F), rhs(F)

  # Time-stepping
  u = Function(V)
  t = dtold  # just to start the iteration

  
  # For the error estimate
  normal = FacetNormal(mesh)
  DG0 = FunctionSpace(mesh, "DG", 0)
  r = TestFunction(DG0)
  residual = r*(-(1/idt)*(u - u_n) + epsilon*div(grad(u)) - dot(w, grad(u)))*dx - (0.5*epsilon)*avg(r)*jump(grad(u),normal)*dS
  cell_residual = Function(DG0)
  

  # list to store t and u
  t_list = []
  u_list = []
  res = []
  dt_list = []
  
  while t<T:
      
      u_D.t = dt  # the boundary condition needs to be updated 

      # Compute solution
      solve(a == L, u, bc)

      # Compute u at the vertices and add them to the list
      u_approx = u.compute_vertex_values(mesh)
      t_list.append(t)
      u_list.append(np.linalg.norm(u_approx))
      #u_list.append(npu_approx)

      # compute the residuals (a posteriori error)
      assemble(residual, tensor=cell_residual.vector())
      res_on_mesh = cell_residual.compute_vertex_values(mesh)
      res.append(res_on_mesh)

      dtnew = dt*(dt_tol/abs(np.mean(res_on_mesh)))**(1/3)
      
    
      dt = dtnew
      dt_list.append(dt)

      # Update current time
      t += dt  # this just add successive terms, a kind of cumulative time

      t = round(t,2) #avoid numerical errors
      idt.assign(dt)   # I change dt to Fenics constant

      
      # Update previous solution
      u_n.assign(u)
      
  
  if plot_u_t==True:
      # Plot the norm of u against t
      fig, ax = plt.subplots(1, 3,figsize=(10,10))

      ax[0].plot(t_list, dt_list)
      ax[0].set_xlabel('$t$')
      ax[0].set_ylabel('$dt$')
      ax[0].set_title('List of time steps against time')
      
      ax[1].plot(t_list, [abs(np.mean(el)) for el in res], marker='*')
      ax[1].set_xlabel('$t$')
      ax[1].set_ylabel('Mean of errors')
      ax[1].set_yscale('log')
      ax[1].set_title('Mean of errors at each time')

      ax[2].plot(t_list, u_list)
      ax[2].set_xlabel('$t$')
      ax[2].set_ylabel('$||u||_2$')
      ax[2].set_title('$||u||_2$ against time $t$')


      plt.show()

  u_list = np.array(u_list) 

  return u_list, res, dt_list


u_list, res, dt_list= exponential_time_double_glazing(1,
                                    tau=1/20,
                                    epsilon = 1/200,
                                    w = Expression(('2*x[1]*(1-x[0]*x[0])', '-2*x[0]*(1-x[1]*x[1])'), degree=3),
                                    # num_steps = 50,
                                    T = 5.0,
                                    nx = 30,
                                    ny = 30,
                                    dt_tol = 1e-2,
                                    dtold = 1e-4,
                                    k = 1,
                                    plot_u=True,
                                    plot_u_t=True)
