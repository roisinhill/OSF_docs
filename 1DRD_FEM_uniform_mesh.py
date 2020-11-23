## 1DRD_FEM_uniform_mesh.py
## FEniCS version: 2019.2.0.dev0
"""
Solve the singularly perturbed differential equation (SPDE)
 -epsilon^2 u" + u =  f = 1-x, for x in (0,1)  with  u(0) = u(1) = 0
on a uniform mesh.
This code is part of
Generating layer-adapted meshes using mesh partial differential equations,
by Róisín Hill and Niall Madden. DOI: 10.17605/OSF.IO/DPEXH
Contact: Róisín Hill <Roisin.Hill@NUIGalway.ie>
"""

from fenics import *

# Define problem parameters
epsilon = 1E-2           # perturbation factor
N = 128                  # mesh intervals

# Create mesh and function space with linear element on that mesh
mesh = UnitIntervalMesh(N)
V = FunctionSpace(mesh, 'P', 1)

u = TrialFunction(V)     # Trial function on V
v = TestFunction(V)      # Test function on V
uN = Function(V)         # Numerical solution in V

# Define the Dirichlet boundary conditions
g = Expression('x[0]*(1-x[0])', degree=2)
bc = DirichletBC(V, g, 'on_boundary')

# Define right hand side of the SPDE, and the weak form
f = Expression ('1-x[0]', degree=2)
a = epsilon*epsilon*inner(grad(u), grad(v))*dx + inner(u,v)*dx
L = f*v*dx
solve(a==L, uN, bc)      # Solve, applying the boundary conditions strongly
