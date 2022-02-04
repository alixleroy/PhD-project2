
from fenics import *
import numpy as np
from vedo.dolfin import plot, Latex, clear, histogram
import matplotlib.pyplot as plt
import pickle 

def solver_para(nx,ny,tau):
  # Create mesh and define function space
  mesh = RectangleMesh(Point(-1, -1), Point(1, 1), nx, ny)
  V = FunctionSpace(mesh, "P", 1)
  
  # Define boundary condition
  tol = 1E-14
  u_D = Expression('near(x[0], 1, tol) ? pow(1-x[1],4)*exp(-t*tau):0', degree=4, tol=tol,tau=tau, t=0)

  def boundary(x, on_boundary):
        return on_boundary

  bc = DirichletBC(V, u_D, boundary)

  # Define initial value
  u_n = project(u_D, V)

  # Define variational problem
  u = TrialFunction(V)
  v = TestFunction(V)
  f = Constant(0.0)

  return(V,mesh,u,v,u_n,f,u_D,bc)

def solver_run(alpha,V,mesh,u,v,u_n,f,u_D,bc,tau, epsilon, num_steps, T):

  dt = T / num_steps # time step size
  w = Expression(('exp(alpha)*2*x[1]*(1-x[0]*x[0])', 'exp(alpha)*-2*x[0]*(1-x[1]*x[1])'),alpha = alpha,degree=3)  #define the wind

  F = u*v*dx + epsilon*dt*dot(grad(u), grad(v))*dx + dt*dot(exp(alpha)*w, grad(u))*v*dx - (u_n + dt*f)*v*dx
  a, L = lhs(F), rhs(F)

  # Time-stepping
  u = Function(V)
  t = 0

  # Create VTK file for saving solution
  vtkfile = File('solution.pvd')

  # list to store t and u
  t_u_list = []

  for n in range(num_steps):

      # Update current time
      t += dt
      u_D.t = t #update the time in the boundary condition

      # Compute solution
      solve(a == L, u, bc)

      # # # Plot solution
      plot(u, cmap='jet', scalarbar='h', text=__doc__)

      # Save to file and plot solution
      vtkfile << (u, t)
      
      # Compute u at the vertices and add them to the list
      u_approx = u.compute_vertex_values(mesh)
      t_u_list.append(u_approx)

      # Update previous solution
      u_n.assign(u)

  interactive()
  return(np.array(t_u_list))

## Run the metropolis hasting algorithm

alpha0 = 1 # Initial value of alpha
iterations = 10 # Lenght of the MCMC chain

tau = 1/10 #Value of tau
epsilon = 1/200 #
num_steps = 100 #number of time steps taken
T = 5.0 #final time 
nx = 30 #size of the grid
ny = 30 #size of the grid

sigma_q = 0.5 #variance of the guess proposal
sigma_p = 1 #variance of the prior
mu_p = 0 #mean of the proposal 
sigma_l = 0.2 #variance of the likelihood
        

## Generate the observed data
noise_star = 1 #noise added to the data
alpha_star = 0
nx_star=30
ny_star=30 
Vmesh,mesh,u,v,u_n,f,u_D,bc= solver_para(nx_star,ny_star,tau)
u_star = solver_run(alpha_star,Vmesh,mesh,u,v,u_n,f,u_D,bc,tau, epsilon, num_steps, T)
# y_star = u_star + np.random.normal(np.zeros(np.shape(u_star)), noise_star)

#u_star = solver_run(3,Vmesh,mesh,u,v,u_n,f,u_D,bc,tau, epsilon, num_steps, T)
