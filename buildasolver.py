#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 11:11:28 2022

@author: s2133976
"""
from fenics import *
import numpy as np
import matplotlib.pyplot

T = 2.0            # final time
num_steps = 20     # number of time steps
dt = T / num_steps # time step size
epsilon = 1/200

# Create mesh and define function space
nx = ny = 8
mesh = UnitSquareMesh(nx, ny)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition
u_D = Expression('x[0]==1 ? 1:0',degree=1)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# Define initial value
#u_n = interpolate(u_D, V)
u_n = project(u_D, V) 


# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(1)

# code the vector field
w=Constant(1)  #np.array([1,1])
w = Expression(('2*x[1]*(1-x[0]*x[0])','-2*x[0]*(1-x[1]*x[1])'),degree=3)
#w = Expression(('-20','-40'),degree=1)
F = u*v*dx + dt*dot(grad(u), grad(v))*dx + dt*dot(w,grad(u))*v*dx - (u_n + dt*f)*v*dx
a, L = lhs(F), rhs(F)

# Time-stepping
u = Function(V)
t = 0
for n in range(num_steps):

    # Update current time
    t += dt
#     u_D.t = t #does not depend on t 

    # Compute solution
    print(n)
    solve(a == L, u, bc)
    # Plot solution
    plot(u)

#     # Compute error at vertices
#     u_e = interpolate(u_D, V)
#     error = np.abs(u_e.vector().array() - u.vector().array()).max()
#     print('t = %.2f: error = %.3g' % (t, error))

    # Update previous solution
    u_n.assign(u)

# Hold plot
#plot(u)
#interactive()