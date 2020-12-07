## 1D_layer_adapted_mesh_MPDE_h_refinement.py
## FEniCS version: 2019.2.0.dev0
'''
Generate 1-sided Bakhvalov-type mesh, for a singularly perturbed ODE
whose solution has a layer near x = 0, by solving the MPDE
   -(rho(x) x'(xi))' = 0 for xi in (0,1) with x(0) = 0 and x(1) = 1,
using uniform h-refinement.

Note: selection of Gauss-Lobatto quadrature rule is executed in lines 22-30.

This code is part of
Generating layer-adapted meshes using mesh partial differential equations,
by Róisín Hill and Niall Madden. DOI: 10.17605/OSF.IO/DPEXH
Contact: Róisín Hill <Roisin.Hill@NUIGalway.ie>
'''

from fenics import *
import matplotlib.pyplot as plt
import math
import numpy as np

# Change quadrature rule to Gauss Lobatto
from FIAT.reference_element import *
from FIAT.quadrature import *
from FIAT.quadrature_schemes import create_quadrature
def create_GaussLobatto_quadrature(ref_el, degree, scheme="default"):
    if degree < 3: degree = 3
    return GaussLobattoLegendreQuadratureLineRule(ref_el, degree)
import FIAT
FIAT.create_quadrature = create_GaussLobatto_quadrature

# Problem parameters
epsilon = 1E-6      # perturbation factor of the physical PDE
N = 32              # mesh intervals in final mesh

# parameters for rho(x)
sigma = 2.5         # related to degree of elements
b = 0.99            # lower bound on reaction coefficient
q = 0.5             # proportion of mesh points in layer
K = q/(sigma*(1-q))

# Parameters for h-refinement steps
N_start = 4         # initial number of mesh intervals
step_index = 0      # initial step number
N_steps = 2**np.arange(round(math.log(N_start,2)),int(math.log(N,2)))

f = Expression('0.0', degree=2)    # Right-and side of MPDE
x_0 = Expression('x[0]', degree=2) # Initial solution

residual_norm_TOL = 0.4   # Stopping criterion tolerance
residual_norm = residual_norm_TOL+1

# Lists for meshes and computational function spaces
mesh_list = []
V_list = []
list_length = (N_steps.size+1)

for i in range(0,list_length):
    mesh_list.append("mesh%2d" %(i))
    V_list.append("V2D%2d" %(i))

# Computactional function space parameters
def comp_space(N):
    meshc = UnitIntervalMesh(N)
    V = FunctionSpace(meshc, 'P', 1) # Function space with P1-elements
    v = TestFunction(V)              # Test function on V
    x = TrialFunction(V)             # Trial function on V
    return meshc, V, v, x

# FEM problem for the MPDE
def MPDE_FEM(rho, xN, x, v, f, meshz, V):
    bc = DirichletBC(V, x_0, 'on_boundary') # Dirichlet boundary condition
    a = rho(xN)*inner(grad(x),grad(v))*dx(meshz) # Left side of weak form
    L = f*v*dx                       # Right side of weak form
    xN = Function(V)                 # Numerical solution in V
    solve(a==L, xN, bc)              # Solve, applying the BCs strongly
    return xN

# Define rho(x) for MPDE
def rho(xN):
    return conditional((K*(b/epsilon)*exp(-b*xN/(sigma*epsilon)))>1.0, \
                           K*(b/epsilon)*exp(-b*xN/(sigma*epsilon)), 1.0)

# The norm of the residual to use as stopping criterion
def calc_residual(rho, V, meshN):
    residual = assemble((rho(xN)*xN.dx(0)*v.dx(0))*dx) # residual of xN
    bcr = DirichletBC(V, '0.0', 'on_boundary')
    bcr.apply(residual)                              # apply Dirichlet BCs
    residual_func = Function(V)
    residual_func.vector()[:] = residual.get_local() # Residual fn on V
    residual_norm = norm(residual_func, 'L2', meshN) # L2-norm of residual
    return residual_norm

# Initial function space parameters
mesh_list[0], V_list[0], v, x = comp_space(N_start)
xN = interpolate(x_0, V_list[0])                     # Initial value for xN

# Iterate through uniform h-refinements
for N_step in N_steps:
    for i in range(0,3):    # Calculate solution three times on each mesh size
         xN = MPDE_FEM(rho, xN, x, v, f, mesh_list[step_index], \
                               V_list[step_index])

    step_index = round(math.log(N_step*2/N_start,2))
    # Generate computational function space on the finer mesh
    mesh_list[step_index], V_list[step_index], v, x = comp_space(N_step*2)

    # Interpolate the solution onto the new computational function space
    xN = interpolate(xN, V_list[step_index])

# Calculate solution and L2 norm of the residual on final function space
while residual_norm > residual_norm_TOL:
    xN = MPDE_FEM(rho, xN, x, v, f, mesh_list[step_index], V_list[step_index])
    residual_norm = calc_residual(rho, V_list[step_index], \
                               mesh_list[step_index])

# Generate the physical mesh
meshp = UnitIntervalMesh(N)
meshp.coordinates()[:,0]  = xN.compute_vertex_values()[:]

# DEFINE REACTION-DIFFUSION PROBLEM TO SOLVE ON MESH GENERATED
# Solve -epsilon^2 u" + u = 1-x for x in (0,1) with u(0) = u(1) = 0
# Physical function space parameters
Vp = FunctionSpace(meshp, 'P', 1)            # Function space with P1-elements
vp = TestFunction(Vp)                        # Test function on V
u = TrialFunction(Vp)                        # Trial function on V
gp = Expression('x[0]*(1-x[0])', degree=2)   # Boundary values
bcp = DirichletBC(Vp, gp, 'on_boundary')     # Dirichlet boundary conditions
fp = Expression('1.0-x[0]', degree=2)        # RHS of PDE

# FEM problem for the reaction-diffusion equation
a = epsilon*epsilon*dot(grad(u), grad(vp))*dx + u*vp*dx
L = fp*vp*dx
uN = Function(Vp)
solve(a==L, uN, bcp)
plot(uN), plt.show()
