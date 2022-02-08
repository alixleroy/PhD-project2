from fenics import *
import numpy as np

def time_double_glazing_smooth(tau, epsilon, num_steps, T, nx, ny, k, alpha):

  dt = T / num_steps # time step size

  # Create mesh and define function space
  mesh = RectangleMesh(Point(-1, -1), Point(1, 1), nx, ny)
  V = FunctionSpace(mesh, "P", k)

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

  # Define wind field
  w = Expression(('2*x[1]*(1-x[0]*x[0])', '-2*x[0]*(1-x[1]*x[1])'), degree=3)

  F = u*v*dx + epsilon*dt*dot(grad(u), grad(v))*dx + dt*dot(exp(alpha)*w, grad(u))*v*dx - (u_n + dt*f)*v*dx
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


if __name__ == "__main__":

    def main():

        tau = 1/10
        epsilon = 1/200
        num_steps = 10
        T = 5.0
        nx = 30
        ny = 30
        k = 1
        alpha = 3

        return time_double_glazing_smooth(tau, epsilon, num_steps, T, nx, ny, k, alpha)

    u = main()
    print(np.shape(u))



