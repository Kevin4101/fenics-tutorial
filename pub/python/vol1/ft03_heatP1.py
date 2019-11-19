"""
FEniCS tutorial demo program: Heat equation with Dirichlet conditions.
Test problem is chosen to give an exact solution at all nodes of the mesh.

  u'= Laplace(u) + f  in the unit square
  u = u_D             on the boundary
  u = u_0             at t = 0

  u = 1 + x^2 + alpha*y^2 + \beta*t
  f = beta - 2 - 2*alpha
"""

from __future__ import print_function
from fenics import *
import numpy as np

T = 0.0            # final time
num_steps = 1     # number of time steps
dt = T / num_steps # time step size
alpha = 0          # parameter alpha
beta = 0         # parameter beta

# Create mesh and define function space
mesh = RectangleMesh(Point(0, 0), Point(0.03,0.08), 30,80) #2D mesh
V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition
u_D1 = Expression('300 + 0*x[0]*x[0] + alpha*x[1]*x[1] + beta*t',
                 degree=2, alpha=alpha, beta=beta, t=0)
u_D2 = Expression('310 + 0*x[0]*x[0] + alpha*x[1]*x[1] + beta*t',
                 degree=2, alpha=alpha, beta=beta, t=0)

def lboundary(y=0, on_boundary):
    return on_lboundary
  
def uboundary(y=0.08, on_boundary):
    return on_uboundary

bc1 = DirichletBC(V, u_D1, lboundary)
bc2 = DirichletBC(V, u_D2, uboundary)

# Define initial value
u_n = interpolate(u_D, V)
#u_n = project(u_D, V)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(beta - 2 - 2*alpha)

F = u*v*dx + dt*dot(grad(u), grad(v))*dx - (u_n + dt*f)*v*dx
a, L = lhs(F), rhs(F)

# Time-stepping
u = Function(V)
t = 0
for n in range(num_steps):

    # Update current time
    t += dt
    u_D.t = t

    # Compute solution
    solve(a == L, u, bc1,bc2)

    # Plot solution
    plot(u)

    # Compute error at vertices
    u_e = interpolate(u_D, V)
    error = np.abs(u_e.vector().array() - u.vector().array()).max()
    print('t = %.2f: error = %.3g' % (t, error))

    # Update previous solution
    u_n.assign(u)

# Hold plot
interactive()
