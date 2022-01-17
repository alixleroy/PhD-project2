"""Double-glazing problem with Dirichlet conditions.
  -epsilon*Laplace(u) + dot(grad(u), w)  = 0    in the square (-1, 1)*(-1, 1)
   u(-1, y) = 0, u(1, y) = 1, u(x, -1) = 0, u(x, 1) = 0
"""

import numpy as np
from fenics import *
from vedo.dolfin import plot, Latex, clear, histogram

def convection_diffusion(epsilon, w, nx, ny, k):
    '''
    epsilon: the viscous parameter
    w: the wind parameter, a vector function or constant
    nx, ny: the number of triangles
    k: order of approximating polynomial
    '''
    # Create mesh and define function space
    mesh = RectangleMesh(Point(-1, -1), Point(1, 1), nx, ny)
    V = FunctionSpace(mesh, "P", k)
    
    # Define boundary condition
    u_L1 = Expression('0', degree=0)
    def boundary_L1(x, on_boundary):
        tol = 1E-14
        return on_boundary and near(x[0], -1, tol)

    bc_L1 = DirichletBC(V, u_L1, boundary_L1)


    u_L2 = Expression('1', degree=0)
    def boundary_L2(x, on_boundary):
        tol = 1E-14
        return on_boundary and near(x[0], 1, tol)

    bc_L2 = DirichletBC(V, u_L2, boundary_L2)

    u_R1 = Expression('0', degree=0)
    def boundary_R1(x, on_boundary):
       tol = 1E-14
       return on_boundary and near(x[1], -1, tol)

    bc_R1 = DirichletBC(V, u_R1, boundary_R1)

    u_R2 = Expression('0', degree=0)
    def boundary_R2(x, on_boundary):
       tol = 1E-14
       return on_boundary and near(x[1], 1, tol)

    bc_R2 = DirichletBC(V, u_R2, boundary_R2)

    bcs = [bc_L1, bc_L2, bc_R1, bc_R2]

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(0)

    # Define the viscous term
    epsilon = Constant(epsilon)
    
    # Define the bilinear and linear forms
    eq = (epsilon*dot(grad(u), grad(v)) + dot(w, grad(u))*v)*dx - f*v*dx
    a = lhs(eq)
    L = rhs(eq)

    # Compute solution
    u = Function(V)
    solve(a == L, u, bcs)
    
    # Plot the solution
    plot(u, cmap='jet', scalarbar='h', text=__doc__)
    
    # Get the values of u at each vertex
    vertex_values_u = u.compute_vertex_values(mesh)
    print(vertex_values_u)


convection_diffusion(epsilon = 1/200,
                     w = Expression(('2*x[1]*(1-x[0]*x[0])', '-2*x[0]*(1-x[1]*x[1])'), degree=3),
                     nx = 50,
                     ny = 50,
                     k = 1
                    )