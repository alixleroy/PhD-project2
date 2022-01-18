"""Double-glazing problem with Dirichlet conditions.
  du/dt - epsilon*Laplace(u) + dot(grad(u), w) = 0 in the square (-1, 1)*(-1, 1)
  and time domain (0, T]
  u(-1, y) = 0, u(1, y) = 1, u(x, -1) = 0, u(x, 1) = 0

"""

from fenics import *
import numpy as np
from vedo.dolfin import plot, Latex, clear, histogram
import matplotlib.pyplot as plt

def time_double_glazing(epsilon, w, num_steps, T, nx, ny, k):
  '''
    epsilon: the viscous parameter
    w: the wind parameter, a vector function or constant
    nx, ny: the number of triangles
    k: order of approximating polynomial
    T: Final time step
    num_steps: number of steps
  '''
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

      # Compute solution
      solve(a == L, u, bc)

      # Plot solution
      # plot(u, cmap='jet', scalarbar='h', text=__doc__)

      # Compute u at the vertices and add them to the list
      u_approx = u.compute_vertex_values(mesh)
      t_u_list.append((t, u_approx))

      # Update previous solution
      u_n.assign(u)

  print(t_u_list)
  
  # Plot the norm of u against t
  fig, ax = plt.subplots(figsize=(10,10))
  t_list = [t_u_list[i][0] for i in range(len(t_u_list))]
  u_norm_list = [np.linalg.norm(t_u_list[i][1]) for i in range(len(t_u_list))]

  ax.plot(t_list, u_norm_list)
  ax.set_xlabel('$t$', fontsize=12)
  ax.set_ylabel('$||u||_2$', fontsize=12)
  ax.set_title('$||u||_2$ against time $t$', fontsize=14)

  plt.show()

# Call the function
time_double_glazing(epsilon = 1/200,
                    w = Expression(('2*x[1]*(1-x[0]*x[0])', '-2*x[0]*(1-x[1]*x[1])'), degree=3),
                    num_steps = 10,
                    T = 5.0,
                    nx = 30,
                    ny = 30,
                    k = 1
                    )
