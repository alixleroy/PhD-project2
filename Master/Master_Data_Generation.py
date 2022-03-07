'''
Generating high-resolution data from double-glazing solver for use in likelihood functions.
'''

import numpy as np
from fenics import *
#import warnings

from Time_Stepping import next_dt

# To silent unwanted output
# logging.getLogger('FFC').setLevel(logging.WARNING)
# warnings.filterwarnings('ignore')
# set_log_active(False)

def master_gen_solver(#nx,
                      #ny,
                      mesh,
                      tau,
                      alpha,
                      epsilon,
                      num_steps,
                      #dt_min,
                      #dt_max,
                      #reg
                      ):

  mesh = refine(mesh)
  #mesh = refine(mesh1)

  # Define function space.
  V = FunctionSpace(mesh, "P", 1)

  # Define boundary condition.
  tol = 1E-14
  u_D = Expression('near(x[0], 1, tol) ? (1-pow(x[1], 4))*(1-exp(-t / tau)):0', degree=4, tol=tol, tau=tau, t=0)

  def boundary(x, on_boundary):
        return on_boundary

  bc = DirichletBC(V, u_D, boundary)

  # Define initial value.
  u_n = project(u_D, V)

  # Define variational problem.
  u = TrialFunction(V)
  v = TestFunction(V)
  f = Constant(0.0)

  # Initialise the time step.
  dt = 0.00001
  dt_f = Constant(dt)

  # Define the wind.
  w = Expression(('2*x[1]*(1-x[0]*x[0])', '-2*x[0]*(1-x[1]*x[1])'), alpha = alpha, degree=3)

  F = u * v * dx + epsilon * dt_f * dot(grad(u), grad(v)) * dx + dt_f * exp(alpha) * dot(w, grad(u)) * v * dx - (u_n + dt_f * f) * v * dx
  a, L = lhs(F), rhs(F)

  # Time-stepping
  u = Function(V)
  t = 0

  # Lists to store t and u.
  u_list = []
  t_list = []

  for _ in range(num_steps):

      # Compute solution.
      solve(a == L, u, bc)

      # Compute u at the vertices and add them to the list.
      # u_approx = u.compute_vertex_values(mesh)
      u_list.append(u.copy(deepcopy=True))

      # Update previous solution.
      u_n.assign(u)
  
      # Time.
      t_list.append(t)

      # Update current time
      # dt = next_dt(dt_min, dt_max, t, tau, reg)
      t += dt
      t = round(t, 10) # avoid numerical errors
      #dt_f.assign(dt)

      u_D.t = t #update the time in the boundary condition

      print(_)

  return u_list, t_list

if __name__ == '__main__':

  from vedo.dolfin import plot, Latex, clear, histogram
  import matplotlib.pyplot as plt
  import pickle

  from Stretch_Mesh import stretch_mesh

  def main():

    # Coarse.
    tau = 1/10
    epsilon = 1/100
    num_steps = 150000
    nx = 32
    ny = 32
    alpha = 0

    # Time stepping.
    dt_min = 1e-3
    dt_max = 0.1
    reg = 10

    mesh = stretch_mesh(nx=nx, ny=ny)

    return master_gen_solver(#nx,
                             #ny,
                             mesh,
                             tau,
                             alpha,
                             epsilon,
                             num_steps,
                             #dt_min,
                             #dt_max,
                             #reg
                             )

  u_list, t_list = main()

  print(np.shape(u_list))

  ### SAVING FILES ###
  index = [2,4,8,16,32,64,128]

  for i in range(6):

    mesh_coarse = stretch_mesh(nx=index[i], ny=index[i])
    V_coarse = FunctionSpace(mesh_coarse, "P", 1)
    u_coarse_project_list     = []
    u_coarse_interpolate_list = []

    for u in u_list:

      u_coarse_project = project(u, V_coarse)
      u_approx_project = u_coarse_project.compute_vertex_values(mesh_coarse)
      u_coarse_project_list.append(u_approx_project.copy())

      u_coarse_interpolate = interpolate(u, V_coarse)
      u_approx_interpolate = u_coarse_interpolate.compute_vertex_values(mesh_coarse)
      u_coarse_interpolate_list.append(u_approx_interpolate.copy())
    
    with open(f'Master/Data_eps100_num150000_tau10/master_data_{index[i]}_project', 'wb') as file:
      pickle.dump(u_coarse_project_list, file)
    with open(f'Master/Data_eps100_num150000_tau10/master_data_{index[i]}_interpolate', 'wb') as file:
      pickle.dump(u_coarse_interpolate_list, file)
  
  # mesh_coarse_128 = stretch_mesh(nx=128, ny=128)
  # mesh_coarse_64  = stretch_mesh(nx=64,  ny=64)
  # mesh_coarse_32  = stretch_mesh(nx=32,  ny=32)
  # mesh_coarse_16  = stretch_mesh(nx=16,  ny=16)
  # mesh_coarse_8   = stretch_mesh(nx=8,   ny=8)
  # mesh_coarse_4   = stretch_mesh(nx=4,   ny=4)
  # mesh_coarse_2   = stretch_mesh(nx=2,   ny=2)

  # V_coarse_128 = FunctionSpace(mesh_coarse_128, "P", 1)
  # V_coarse_64  = FunctionSpace(mesh_coarse_64,  "P", 1)
  # V_coarse_32  = FunctionSpace(mesh_coarse_32,  "P", 1)
  # V_coarse_16  = FunctionSpace(mesh_coarse_16,  "P", 1)
  # V_coarse_8   = FunctionSpace(mesh_coarse_8,   "P", 1)
  # V_coarse_4   = FunctionSpace(mesh_coarse_4,   "P", 1)
  # V_coarse_2   = FunctionSpace(mesh_coarse_2,   "P", 1)

  # u_coarse_128_project_list = []
  # u_coarse_64_project_list  = []
  # u_coarse_32_project_list  = []
  # u_coarse_16_project_list  = []
  # u_coarse_8_project_list   = []
  # u_coarse_4_project_list   = []
  # u_coarse_2_project_list   = []

  # u_coarse_128_interpolate_list = []
  # u_coarse_64_interpolate_list  = []
  # u_coarse_32_interpolate_list  = []
  # u_coarse_16_interpolate_list  = []
  # u_coarse_8_interpolate_list   = []
  # u_coarse_4_interpolate_list   = []
  # u_coarse_2_interpolate_list   = []

  # for u in u_list:
    
  #   u_coarse_128_project = project(u, V_coarse_128)
  #   u_approx_128_project = u_coarse_128_project.compute_vertex_values(mesh_coarse_128)
  #   u_coarse_128_project_list.append(u_approx_128_project.copy())
  #   u_coarse_128_interpolate = interpolate(u, V_coarse_128)
  #   u_approx_128_interpolate = u_coarse_128_interpolate.compute_vertex_values(mesh_coarse_128)
  #   u_coarse_128_interpolate_list.append(u_approx_128_interpolate.copy())

  #   u_coarse_64_project = project(u, V_coarse_64)
  #   u_approx_64_project = u_coarse_64_project.compute_vertex_values(mesh_coarse_64)
  #   u_coarse_64_project_list.append(u_approx_64_project.copy())
  #   u_coarse_64_interpolate = interpolate(u, V_coarse_64)
  #   u_approx_64_interpolate = u_coarse_64_interpolate.compute_vertex_values(mesh_coarse_64)
  #   u_coarse_64_interpolate_list.append(u_approx_64_interpolate.copy())

  #   u_coarse_32_project = project(u, V_coarse_32)
  #   u_approx_32_project = u_coarse_32_project.compute_vertex_values(mesh_coarse_32)
  #   u_coarse_32_project_list.append(u_approx_32_project.copy())
  #   u_coarse_32_interpolate = interpolate(u, V_coarse_32)
  #   u_approx_32_interpolate = u_coarse_32_interpolate.compute_vertex_values(mesh_coarse_32)
  #   u_coarse_32_interpolate_list.append(u_approx_32_interpolate.copy())

  #   u_coarse_16_project = project(u, V_coarse_16)
  #   u_approx_16_project = u_coarse_16_project.compute_vertex_values(mesh_coarse_16)
  #   u_coarse_16_project_list.append(u_approx_16_project.copy())
  #   u_coarse_16_interpolate = interpolate(u, V_coarse_16)
  #   u_approx_16_interpolate = u_coarse_16_interpolate.compute_vertex_values(mesh_coarse_16)
  #   u_coarse_16_interpolate_list.append(u_approx_16_interpolate.copy())

  #   u_coarse_8_project = project(u, V_coarse_8)
  #   u_approx_8_project = u_coarse_8_project.compute_vertex_values(mesh_coarse_8)
  #   u_coarse_8_project_list.append(u_approx_8_project.copy())
  #   u_coarse_8_interpolate = interpolate(u, V_coarse_8)
  #   u_approx_8_interpolate = u_coarse_8_interpolate.compute_vertex_values(mesh_coarse_8)
  #   u_coarse_8_interpolate_list.append(u_approx_8_interpolate.copy())

  #   u_coarse_4_project = project(u, V_coarse_4)
  #   u_approx_4_project = u_coarse_4_project.compute_vertex_values(mesh_coarse_4)
  #   u_coarse_4_project_list.append(u_approx_4_project.copy())
  #   u_coarse_4_interpolate = interpolate(u, V_coarse_4)
  #   u_approx_4_interpolate = u_coarse_4_interpolate.compute_vertex_values(mesh_coarse_4)
  #   u_coarse_4_interpolate_list.append(u_approx_4_interpolate.copy())

  #   u_coarse_2_project = project(u, V_coarse_2)
  #   u_approx_2_project = u_coarse_2_project.compute_vertex_values(mesh_coarse_2)
  #   u_coarse_2_project_list.append(u_approx_2_project.copy())
  #   u_coarse_2_interpolate = interpolate(u, V_coarse_2)
  #   u_approx_2_interpolate = u_coarse_2_interpolate.compute_vertex_values(mesh_coarse_2)
  #   u_coarse_2_interpolate_list.append(u_approx_2_interpolate.copy())

  # with open('Master/Data/master_data_128_project', 'wb') as file:
  #   pickle.dump(u_coarse_128_project_list, file)
  # with open('Master/Data/master_data_128_interpolate', 'wb') as file:
  #   pickle.dump(u_coarse_128_interpolate_list, file)

  # with open('Master/Data/master_data_64_project', 'wb') as file:
  #   pickle.dump(u_coarse_64_project_list, file)
  # with open('Master/Data/master_data_64_interpolate', 'wb') as file:
  #   pickle.dump(u_coarse_64_interpolate_list, file)

  # with open('Master/Data/master_data_32_project', 'wb') as file:
  #   pickle.dump(u_coarse_32_project_list, file)
  # with open('Master/Data/master_data_32_interpolate', 'wb') as file:
  #   pickle.dump(u_coarse_32_interpolate_list, file)

  # with open('Master/Data/master_data_16_project', 'wb') as file:
  #   pickle.dump(u_coarse_16_project_list, file)
  # with open('Master/Data/master_data_16_interpolate', 'wb') as file:
  #   pickle.dump(u_coarse_16_interpolate_list, file)

  # with open('Master/Data/master_data_8_project', 'wb') as file:
  #   pickle.dump(u_coarse_8_project_list, file)
  # with open('Master/Data/master_data_8_interpolate', 'wb') as file:
  #   pickle.dump(u_coarse_8_interpolate_list, file)
  
  # with open('Master/Data/master_data_4_project', 'wb') as file:
  #   pickle.dump(u_coarse_4_project_list, file)
  # with open('Master/Data/master_data_4_interpolate', 'wb') as file:
  #   pickle.dump(u_coarse_4_interpolate_list, file)
  
  # with open('Master/Data/master_data_2_project', 'wb') as file:
  #   pickle.dump(u_coarse_2_project_list, file)
  # with open('Master/Data/master_data_2_interpolate', 'wb') as file:
  #   pickle.dump(u_coarse_2_interpolate_list, file)