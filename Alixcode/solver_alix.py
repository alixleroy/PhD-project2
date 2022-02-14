
from fenics import *
import numpy as np
from vedo.dolfin import plot, Latex, clear, histogram
import matplotlib.pyplot as plt
import pickle 
import time

def solver(nx,ny,tau,parameter, epsilon, dt, T,w):

  # Create mesh and define function space
  mesh = RectangleMesh(Point(-1, -1), Point(1, 1), nx, ny)
  V = FunctionSpace(mesh, "P", 1)

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
  

  F = u*v*dx + epsilon*dt*dot(grad(u), grad(v))*dx + dt*exp(parameter)*dot(w, grad(u))*v*dx - (u_n + dt*f)*v*dx
  a, L = lhs(F), rhs(F)

  # Time-stepping
  u = Function(V)
  t = 0

  # list to store t and u
  u_list = []
  unorm_list = []
  t_list = []

  # get the right dt in the list dt_list, get the value of increment 
  while t<T:
      
      # Compute solution
      solve(a == L, u, bc)

      # Compute u at the vertices and add them to the list
      u_approx = u.compute_vertex_values(mesh)
      u_list.append(u_approx)

      unorm_list.append(np.linalg.norm(u_approx))


      # Update previous solution
      u_n.assign(u)
      t_list.append(t)

      # Update current time
      t += dt 
      u_D.t = t #update the time in the boundary condition

  
  u_list = np.array(u_list)
  unorm_list = np.array(unorm_list)
  t_list = np.array(t_list)
  
  return(u_list, unorm_list, t_list)

def solver_ada(nx,ny,tau,parameter, epsilon, dt_list, T,w):

  # Create mesh and define function space
  mesh = RectangleMesh(Point(-1, -1), Point(1, 1), nx, ny)
  V = FunctionSpace(mesh, "P", 1)

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

  ## initialise the time step
  dt = dt_list[0]
  idt = Constant(dt)

  F = u*v*dx + epsilon*idt*dot(grad(u), grad(v))*dx + idt*exp(parameter)*dot(w, grad(u))*v*dx - (u_n + idt*f)*v*dx
  a, L = lhs(F), rhs(F)

  # Time-stepping
  u = Function(V)
  t = 0

  # list to store t and u
  u_list = []
  unorm_list = []
  t_list = []

  i=0
  while t<T:
      i=i+1
      # Compute solution
      solve(a == L, u, bc)

      # Compute u at the vertices and add them to the list
      u_approx = u.compute_vertex_values(mesh)
      u_list.append(u_approx)

      unorm_list.append(np.linalg.norm(u_approx))


      # Update previous solution
      u_n.assign(u)
  
      # Time 
      t_list.append(t)

      # Update current time
      dt +=dt_list[i]
      t += dt
      idt.assign(dt)

      u_D.t = t #update the time in the boundary condition


  u_list = np.array(u_list)
  unorm_list = np.array(unorm_list)
  t_list = np.array(t_list)

  return(u_list, unorm_list, t_list)

def solver_para(nx,ny):
  # Create mesh and define function space
  mesh = RectangleMesh(Point(-1, -1), Point(1, 1), nx, ny)
  V = FunctionSpace(mesh, "P", 1)

  # Define variational problem
  u = TrialFunction(V)
  v = TestFunction(V)
  f = Constant(0)  
  return( mesh, V, u, v, f)

def solver_loop(parameter,mesh,V,u,v,f,tau, epsilon, num_steps, T,w):


  dt = 1 / num_steps # time step size

  # Define boundary condition
  tol = 1E-14
  u_D = Expression('near(x[0], 1, tol) ? (1-pow(x[1], 4))*(1-exp(-t / tau)):0', degree=4, tol=tol, tau=tau, t=0)

  def boundary(x, on_boundary):
        return on_boundary

  bc = DirichletBC(V, u_D, boundary)

  # Define initial value
  u_n = project(u_D, V)

  # Define the linear and bilinear forms
  F = u*v*dx + epsilon*dt*dot(grad(u), grad(v))*dx + dt*exp(parameter)*dot(w, grad(u))*v*dx - (u_n + dt*f)*v*dx
  a, L = lhs(F), rhs(F)

  # Time-stepping
  u = Function(V)
  t = 0

  #list to save values 
  t_list = []
  u_list =[]
  unorm_list = []
  while t<T: 


      # Compute solution
      solve(a == L, u, bc)

      # Compute u at the vertices and add them to the list
      u_approx = u.compute_vertex_values(mesh)
      u_list.append(u_approx)

      unorm_list.append(np.linalg.norm(u_approx))


      # Update previous solution
      u_n.assign(u)

      # Update current time
      t_list.append(t)
      t += dt
      u_D.t = t #update the time in the boundary condition
  
  u_list = np.array(u_list)
  unorm_list = np.array(unorm_list)
  t_list = np.array(t_list)
  
  return(u_list, unorm_list, t_list)

# def solver_loop_ada(parameter,mesh,V,u,v,f,tau, epsilon, dt_max, T,w):


#   # Define boundary condition
#   tol = 1E-14
#   u_D = Expression('near(x[0], 1, tol) ? (1-pow(x[1], 4))*(1-exp(-t / tau)):0', degree=4, tol=tol, tau=tau, t=0)

#   def boundary(x, on_boundary):
#         return on_boundary

#   bc = DirichletBC(V, u_D, boundary)

#   # Define initial value
#   u_n = project(u_D, V)

#   ## initialise the time step
#   dt = dt_max*0.1
#   idt = Constant(dt)

#   # Define the linear and bilinear forms
#   F = u*v*dx + epsilon*dt*dot(grad(u), grad(v))*dx + dt*exp(parameter)*dot(w, grad(u))*v*dx - (u_n + dt*f)*v*dx
#   a, L = lhs(F), rhs(F)

#   # Time-stepping
#   u = Function(V)
#   t = 0

#   #list to save values 
#   t_list = []
#   u_list =[]
#   unorm_list = []

#   ##
#   while t<T:

#       # Compute solution
#       solve(a == L, u, bc)

#       # Compute u at the vertices and add them to the list
#       u_approx = u.compute_vertex_values(mesh)
#       u_list.append(u_approx)

#       unorm_list.append(np.linalg.norm(u_approx))


#       # Update previous solution
#       u_n.assign(u)
  
#       # Update current time
#       t_list.append(t)

#       # Update current time
#       dt +=((1-np.exp(-t/tau))*0.9+0.1)*dt_max
#       t += dt
#       idt.assign(dt)

#       u_D.t = t #update the time in the boundary condition
      
#   u_list = np.array(u_list)
#   unorm_list = np.array(unorm_list)
#   t_list = np.array(t_list)
  
#   return(u_list, unorm_list, t_list)
