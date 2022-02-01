
from fenics import *
import numpy as np
from vedo.dolfin import plot, Latex, clear, histogram
import matplotlib.pyplot as plt
import pickle 

## I - Parameters of the solver 
def solver_parameters(nx,ny,tau,epsilon):
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

    return(mesh,V,u_D,bc,u_n,u,v,f)


# define a function to call. 
def solver_dg(w,num_steps, dt,tau, epsilon,mesh,V,u_D,bc,u_n,u,v,f):

  # defining the PDE
  F = u*v*dx + epsilon*dt*dot(grad(u), grad(v))*dx + dt*dot(w, grad(u))*v*dx - (u_n + dt*f)*v*dx
  a, L = lhs(F), rhs(F)

  # Time-stepping
  u = Function(V)
  t = 0

  # list to store t and u
  t_u_list = []

  for n in range(num_steps):

      # Update current time
      t += dt
      u_D.t = t #update the time in the boundary condition

      # Compute solution
      solve(a == L, u, bc)

      # # Plot solution
      # plot(u, cmap='jet', scalarbar='h', text=__doc__)


      # Compute u at the vertices and add them to the list
      u_approx = u.compute_vertex_values(mesh)
      t_u_list.append(u_approx)


      # Update previous solution
      u_n.assign(u)


  return(np.array(t_u_list))
