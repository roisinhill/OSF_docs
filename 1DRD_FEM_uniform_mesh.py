## 1DRD_FEM_uniform_mesh.py
"""
Solve the singularly perturbed problem
 -epsilon^2 u'' + u =  f = 1-x, for x in (0,1)
with  u(0) = u(1) = 0
on a uniform mesh

This code is part of
Mesh partial differential equations (PDEs): application and implementation,
Wwitten by Roisin Hill and Niall Madden
Contact: Roisin Hill
For: <cite paper>
"""

from fenics import *

# Define problem parameters
epsilon = 1E-2		# perturbation factor
N = 128				# mesh intervals

# Create mesh and function space with linear element on that mesh
mesh = UnitIntervalMesh(N)
V = FunctionSpace(mesh, 'P', 1)

u = TrialFunction(V) # Trial functions on V
v = TestFunction(V)  # Test functions on V
uN = Function(V)     # Numerical solution in V

# Define the Dirichlet boundary conditions
g = Expression('x[0]*(1-x[0])', degree = 2)
bc = DirichletBC(V, g, 'on_boundary')

# Define right hand side of the SPP, and the weak form
f = Expression ('1-x[0]', degree = 2)
a = epsilon*epsilon*inner(grad(u), grad(v))*dx + inner(u,v)*dx
L = f*v*dx
solve(a==L, uN, bc) # Solve, applying the boundary conditions strongly
