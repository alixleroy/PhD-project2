### 1/ Setting up the PDE solver with smooth boundary conditions 

"""Double-glazing problem with smooth conditions.
  du/dt - epsilon*Laplace(u) + dot(grad(u), w) = 0 in the square (-1, 1)*(-1, 1)
  and time domain (0, T]
  u(-1, y) = 0, u(1, y) = (1-y)^4 (1-exp(-tau t)), u(x, -1) = 0, u(x, 1) = 0

"""

from fenics import *
import numpy as np
from vedo.dolfin import plot, Latex, clear, histogram
import matplotlib.pyplot as plt
import pickle 


def time_double_glazing_smooth(tau, epsilon, w, num_steps, T, nx, ny, k):
  '''
    tau: a constant that control the rate at which the wall heats up 
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

  
  # # Plot the norm of u against t
  # fig, ax = plt.subplots(figsize=(10,10))
  # t_list = [t_u_list[i][0] for i in range(len(t_u_list))]
  # u_norm_list = [np.linalg.norm(t_u_list[i][1]) for i in range(len(t_u_list))]

  # ax.plot(t_list, u_norm_list)
  # ax.set_xlabel('$t$', fontsize=12)
  # ax.set_ylabel('$||u||_2$', fontsize=12)
  # ax.set_title('$||u||_2$ against time $t$', fontsize=14)

  # plt.show()


  return(t_u_list)

# time_double_glazing_smooth(tau=1/10,
#                     epsilon = 1/200,
#                     w = Expression(('2*x[1]*(1-x[0]*x[0])', '-2*x[0]*(1-x[1]*x[1])'), degree=3),
#                     num_steps = 10,
#                     T = 5.0,
#                     nx = 30,
#                     ny = 30,
#                     k = 1
#                     )

### 2/ Creating the data calling on the solver up the PDE solver with smooth boundary conditions 

# The wind parameters can be designed in two ways : either we have a wind that is modified with a 
# parameter alpha that varies (from a uniform distribution)

# a) Pertubating the wind by multiplying by random log normal
# gamma_i  = np.random.normal(1,1)
# w = Expression(('gamma_i*2*x[1]*(1-x[0]*x[0])', 'gamma_i*-2*x[0]*(1-x[1]*x[1])'), degree=3,gamma_i=gamma_i),

# Vary the value of the wind 
n_mesh = 30
num_steps0 = 5 # time steps 

M =3 #number of solutions generated 
u_sols_a = []   #np.zeros((num_steps0*M*n_mesh*n_mesh,1))
y_sols_a = []   

for i in range(M): 
  gamma_i  = np.random.lognormal(1,1)
  u_i = np.array(time_double_glazing_smooth(
                    tau = 1,
                    epsilon = 1/200,
                    w = Expression(('gamma_i*2*x[1]*(1-x[0]*x[0])', 'gamma_i*-2*x[0]*(1-x[1]*x[1])'), degree=3,gamma_i=gamma_i),
                    num_steps = num_steps0,
                    T = 2.0,
                    nx = n_mesh,
                    ny = n_mesh,
                    k = 1
                    ))
  ## Returns a (num_steps0, nmesh*(nmesh + 1)+1 ) vectors of solutions
  # Choose a time T = 2.0 at which we analyse the solution and add noise to create observed data
  u_iT = u_i[num_steps0-1,:]
  y_sol = u_i + np.random.random(len(u_iT))

  y_sols_a.extend(list(y_sol))
  u_sols_a.extend(list(u_i)) #do not need to keep the structure of the data

## Save the first condition
smooth_boundary_winda = "smoothboundary_data/u_smooth_boundary_winda-ns="+str(num_steps0)+"-M="+str(M)+".csv"
with open(smooth_boundary_winda, 'wb') as f:
    pickle.dump(u_sols_a, f)

## Save the first condition
smooth_boundary_winda = "smoothboundary_data/y_smooth_boundary_winda-ns="+str(num_steps0)+"-M="+str(M)+".csv"
with open(smooth_boundary_winda, 'wb') as f:
    pickle.dump(y_sols_a, f)


# ## b) Perturbating the wind by adding a divergence free perturbation
# alpha = positive scaling parameter, log normal parameter? 
# w = Expression(('2*x[1]*(1-x[0]*x[0])+alpha*(x[0]+x[1])', '-2*x[0]*(1-x[1]*x[1])+alpha*(x[0]-x[1])'), degree=3,alpha=alpha)

# Vary the value of the wind 
n_mesh = 30
num_steps0 = 5 # time steps 

M =3 #number of solutions generated 
u_sols_a = []   #np.zeros((num_steps0*M*n_mesh*n_mesh,1))
y_sols_a = []   

for i in range(M): 
  alpha  = np.random.lognormal(1,1)
  u_i = np.array(time_double_glazing_smooth(
                    tau = 1,
                    epsilon = 1/200,
                    w = Expression(('2*x[1]*(1-x[0]*x[0])+alpha*(x[0]+x[1])', '-2*x[0]*(1-x[1]*x[1])+alpha*(x[0]-x[1])'), degree=3,alpha=alpha),
                    num_steps = num_steps0,
                    T = 2.0,
                    nx = n_mesh,
                    ny = n_mesh,
                    k = 1
                    ))
  ## Returns a (num_steps0, nmesh*(nmesh + 1)+1 ) vectors of solutions
  # Choose a time T = 2.0 at which we analyse the solution and add noise to create observed data
  u_iT = u_i[num_steps0-1,:]
  y_sol = u_i + np.random.random(len(u_iT))

  y_sols_a.extend(list(y_sol))
  u_sols_a.extend(list(u_i)) #do not need to keep the structure of the data

## Save the first condition
smooth_boundary_winda = "smoothboundary_data/u_smooth_boundary_winda-ns="+str(num_steps0)+"-M="+str(M)+".csv"
with open(smooth_boundary_winda, 'wb') as f:
    pickle.dump(u_sols_a, f)

## Save the first condition
smooth_boundary_winda = "smoothboundary_data/y_smooth_boundary_winda-ns="+str(num_steps0)+"-M="+str(M)+".csv"
with open(smooth_boundary_winda, 'wb') as f:
    pickle.dump(y_sols_a, f)

