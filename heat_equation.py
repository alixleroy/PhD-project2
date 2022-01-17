"""
Heat equation with Dirichlet conditions.
Test problem is chosen to give an exact solution at all nodes of the mesh.
  u'= Laplace(u) + f  in the unit square
  u = u_D             on the boundary
  u = u_0             at t = 0
  u = 1 + x^2 + alpha*y^2 + \beta*t
  f = beta - 2 - 2*alpha
"""

from fenics import *
import numpy as np
from vedo.dolfin import plot, Latex, clear, histogram

def heat_equation(num_steps, T):
  '''
  T: Final time step
  num_steps: number of steps
  '''
  dt = T / num_steps # time step size
  alpha = 3          # parameter alpha
  beta = 1.2         # parameter beta

  # Create mesh and define function space
  nx = ny = 8
  mesh = UnitSquareMesh(nx, ny)
  V = FunctionSpace(mesh, 'P', 1)

  # Define boundary condition
  u_D = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t',
                  degree=2, alpha=alpha, beta=beta, t=0)

  def boundary(x, on_boundary):
      return on_boundary

  bc = DirichletBC(V, u_D, boundary)

  # Define initial value
  u_n = interpolate(u_D, V)
  #u_n = project(u_D, V)

  # Define variational problem
  u = TrialFunction(V)
  v = TestFunction(V)
  f = Constant(beta - 2 - 2*alpha)

  F = u*v*dx + dt*dot(grad(u), grad(v))*dx - (u_n + dt*f)*v*dx
  a, L = lhs(F), rhs(F)

  # Time-stepping
  u = Function(V)
  t = 0
  for n in range(num_steps):

      # Update current time
      t += dt
      u_D.t = t

      # Compute solution
      solve(a == L, u, bc)

      # Plot solution
      plot(u, cmap='jet', scalarbar='h', text=__doc__)

      # Compute error at vertices
      u_e = u_D.compute_vertex_values(mesh)
      u_approx = u.compute_vertex_values(mesh)
      error = np.abs(u_e - u_approx).max()
      print('t = %.2f: error = %.3g' % (t, error))

      # Update previous solution
      u_n.assign(u)

heat_equation(num_steps=5, T=1.0)