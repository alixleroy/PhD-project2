from fenics import *
import numpy as np

from Stretch_Mesh import stretch_mesh

def solver_para(nx,ny,tau):
  # Create mesh and define function space
  mesh = stretch_mesh(nx=nx,ny=ny)
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
  
  return V, u, v, u_n, f, u_D, bc, mesh

def solver_run(alpha, V, u, v, u_n, f, u_D, bc, mesh, epsilon, num_steps):
  
  dt = 0.01 # time step size

  w = Expression(('exp(alpha)*2*x[1]*(1-x[0]*x[0])', 'exp(alpha)*-2*x[0]*(1-x[1]*x[1])'),alpha = alpha,degree=3)  #define the wind

  F = u * v * dx + epsilon * dt * dot(grad(u), grad(v)) * dx + dt * dot(w, grad(u)) * v * dx - (u_n + dt * f) * v * dx
  
  a, L = lhs(F), rhs(F)

  # Time-stepping
  t = 0

  # Create VTK file for saving solution
  # namefile = str(alpha)+"-alpha.pvd"
  # vtkfile = File("Alixcode/alixswork/"+namefile)
  

  # list to store t and u
  u = Function(V)
  u_list = []

  for n in range(num_steps):

      # Update current time
      t += dt
      u_D.t = t #update the time in the boundary condition


      # Compute solution
      solve(a == L, u, bc)

      # # # Plot solution
      # plot(u, cmap='jet', scalarbar='h', text=__doc__)

      # Save to file and plot solution
      # vtkfile << (u, t)
      
      # Compute u at the vertices and add them to the list
      u_approx = u.compute_vertex_values(mesh)
      u_list.append(u_approx.copy())

      # Update previous solution
      u_n.assign(u)
  
  return u_list

if __name__ == "__main__":

    from vedo.dolfin import plot, Latex, clear, histogram
    import matplotlib.pyplot as plt

    def main():

        # Coarse.
        tau = 1/10
        epsilon = 1/200
        num_steps = 100
        nx = 32
        ny = 32
        alpha = 0

        V, u, v, u_n, f, u_D, bc, mesh = solver_para(nx,ny,tau)

        return solver_run(alpha, V, u, v, u_n, f, u_D, bc, mesh, epsilon, num_steps), mesh    

    u_list, mesh = main()

    # mesh_fine = refine(mesh)
    # V_fine = FunctionSpace(mesh_fine, "P", 1)
    # u_fine = project(u_list[0], V_fine)
    # u_approx_fine = u_fine.compute_vertex_values(mesh_fine)
    # u_approx = u_list[0]
    # print(np.shape(u_approx), np.shape(u_approx_fine))