#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 11:11:28 2022

@author: s2133976
"""
from fenics import *
import numpy as np
import matplotlib.pyplot as plt

from vedo.dolfin import plot, Latex, clear, histogram


T = 5.0            # final time
num_steps = 10     # number of time steps
dt = T / num_steps # time step size
epsilon = 1/200

# Create mesh and define function space
nx = ny = 30

#mesh = UnitSquareMesh(nx, ny) #discrepency
mesh = RectangleMesh(Point(-1, -1), Point(1, 1), nx, ny)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition
tol = 1E-14
u_D = Expression('near(x[0], 1, tol)? 1:0',degree=1, tol=tol)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# Define initial value
#u_n = interpolate(u_D, V)
u_n = project(u_D, V) 


# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0)

# code the vector field
#w=Constant(1)  #np.array([1,1])
w = Expression(('2*x[1]*(1-x[0]*x[0])','-2*x[0]*(1-x[1]*x[1])'),degree=3)
#w = Expression(('-20','-40'),degree=1)
F = u*v*dx + epsilon*dt*dot(grad(u), grad(v))*dx + dt*dot(w,grad(u))*v*dx - (u_n + dt*f)*v*dx
a, L = lhs(F), rhs(F)

# Time-stepping
u = Function(V)
t = 0

t_u_list = []

for n in range(num_steps):

    # Update current time
    t += dt
#     u_D.t = t #does not depend on t 

    # Compute solution
    solve(a == L, u, bc)
    # Plot solution
    plot(u, cmap='jet', scalarbar='h', text=__doc__)

#     # Compute error at vertices
#     u_e = interpolate(u_D, V)
#     error = np.abs(u_e.vector().array() - u.vector().array()).max()
#     print('t = %.2f: error = %.3g' % (t, error))
    # Compute u at the vertices and add them to the list
    u_approx = u.compute_vertex_values(mesh)
    t_u_list.append((t, u_approx))

    # Update previous solution
    u_n.assign(u)

# Plot the norm of u against t
fig, ax = plt.subplots(figsize=(10,10))
t_list = [t_u_list[i][0] for i in range(len(t_u_list))]
u_norm_list = [np.linalg.norm(t_u_list[i][1]) for i in range(len(t_u_list))]

ax.plot(t_list, u_norm_list)
ax.set_xlabel('$t$', fontsize=12)
ax.set_ylabel('$||u||_2$', fontsize=12)
ax.set_title('$||u||_2$ against time $t$', fontsize=14)

plt.show()
# Hold plot
#plot(u)
#interactive()

