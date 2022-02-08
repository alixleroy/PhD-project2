"""Poisson equation with Dirichlet conditions.
  -Laplace(u) = f    in the unit square
            u = uD   on the boundary
  uD = 1 + x^2 + 2*y^2
  (f = -6)
"""

from fenics import *
from vedo.dolfin import plot, Latex, clear, histogram
import numpy as np

def poisson():
  # Create mesh and define function space
  mesh = UnitSquareMesh(8, 8)
  V = FunctionSpace(mesh, "P", 1)

  # Define boundary condition
  uD = Expression("1 + x[0]*x[0] + 2*x[1]*x[1]", degree=2)
  bc = DirichletBC(V, uD, "on_boundary")

  # Define variational problem
  u = TrialFunction(V)
  v = TestFunction(V)
  f = Constant(-6.0)
  a = dot(grad(u), grad(v))*dx
  L = f*v*dx

  # Compute solution
  u = Function(V)
  solve(a==L , u, bc)
  plot(u, cmap='jet', scalarbar='h', text=__doc__)

  # Get the values of u at each vertex
  vertex_values_u = u.compute_vertex_values(mesh)
  print(vertex_values_u)

poisson()