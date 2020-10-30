## 1D_layer_adapted_mesh_rh_refinement.py
## FEniCS version: 2019.2.0.dev0
'''
Generate 1-sided Bakhvalov-type mesh, for a singularly perturbed ODE
whose solution has a layer near x = 0, by solving the MPDE
   -(rho(x) x'(xi))' = 0 for xi in (0,1) with x(0) = 0 and x(1) = 1,

Gauss-Lobatto quadrature rule is used:
i.e. replace 'return GaussJacobiQuadratureLineRule(ref_el, m)'
with 'return GaussLobattoLegendreQuadratureLineRule(ref_el, m)'
in the file FIAT/quadrature.py

This code is part of
Generating layer-adapted meshes using mesh partial differential equations,
by Róisín Hill and Niall Madden. DOI:???
Contact: Róisín Hill <Roisin.Hill@NUIGalway.ie>
'''

from fenics import *
import matplotlib.pyplot as plt
import math

# Set degree for the Gauss-Lobatto quadrature rule
parameters['form_compiler']['quadrature_degree'] =  4

# Problem parameters
epsilon = 1E-6      # perturbation factor of the physical PDE
N = 32              # mesh intervals in final mesh

# parameters for rho(x)
sigma = 2.5         # related to degree of elements
b = 0.99            # lower bound on reaction coefficient
q = 0.5             # proportion of mesh points in layer
K = q/(sigma*(1-q))

# Define right side of MPDE and initial solution
f = Expression('0.0', degree=2)
x_0 = Expression('x[0]', degree=2)

# Define stopping criteria tolerance and intiial value
residual_norm_TOL = 0.02
residual_norm = residual_norm_TOL+1

# Computactional function space parameters
def comp_space():
    mesh = UnitIntervalMesh(N)
    V = FunctionSpace(mesh, 'P', 1)  # Function space with P1-elements
    v = TestFunction(V)              # Test function on V
    x = TrialFunction(V)             # Trial function on V
    return mesh, V, v, x

# Define weak form of MMPDE
def weak_form():
    bc = DirichletBC(V, x_0, 'on_boundary')    # Dirichlet boundary condition
    a = rho()*inner(grad(x),grad(v))*dx        # Left side of weak form
    L = f*v*dx                       # Right side of weak form
    xN = Function(V)                 # Numerical solution in V
    solve(a==L, xN, bc)              # Solve, applying the BCs strongly
    return xN

# Define rho(x) for MPDE
def rho():
    return conditional((K*(b/epsilon)*exp(-b*xN/(sigma*epsilon)))>1.0, K*(b/epsilon)*exp(-b*xN/(sigma*epsilon)), 1.0)

# calculate the norm of the residual to use as stopping criterion
def calc_residual():
    residual = assemble((rho()*xN.dx(0)*v.dx(0))*dx) # residual of xN
    bcr = DirichletBC(V, '0.0', 'on_boundary')
    bcr.apply(residual)                              # apply Dirichlet BCs
    residual_func = Function(V)
    residual_func.vector()[:] = residual.get_local() # Residual fn on V
    residual_norm = norm(residual_func, 'L2', mesh)  # L2-norm of residual
    return residual_norm

mesh, V, v, x = comp_space()    # generate nesh and computational function space
xN = interpolate(x_0, V)        # Initial value for xN

# Calculate solution and L2 norm of the residual
iterations=0
while residual_norm > residual_norm_TOL and iterations < N/2.0+5:
    xN = weak_form()
    residual_norm = calc_residual()
    iterations +=1

meshp = UnitIntervalMesh(N)
meshp.coordinates()[:,0] = xN.compute_vertex_values()

# DEFINE REACTION DIFFUSION PROBLEM TO SOLVE ON MESH GENERATED
# Solve -epsilon^2 u" + u = 1-x for x in (0,1) with u(0) = u(1) = 0
# Physical function space parameters
Vp = FunctionSpace(meshp, 'P', 1)            # Function space with P1-elements
vp = TestFunction(Vp)                        # Test function on V
u = TrialFunction(Vp)                        # Trial function on V
gp = Expression('x[0]*(1-x[0])', degree=2)   # Boundary values
bcp = DirichletBC(Vp, gp, 'on_boundary')     # Dirichlet boundary conditions
fp= Expression('1.0-x[0]', degree=2)         # RHS of PDE

# FEM problem for the reaction-diffusion equation
a = epsilon*epsilon*dot(grad(u), grad(vp))*dx + u*vp*dx
L = fp*vp*dx
uN = Function(Vp)
solve(a==L, uN, bcp)
plot(uN), plt.show()
