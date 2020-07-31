## 1D_layer_adapted_mesh_rh_refinement.py
"""
Generate 1 sided Bakhvalov type mesh, for a singularly
perturbed ODE whose solution has a layer near x = 0,
by solving the MPDE -(rho(x) x'(xi))' = 0 for xi in (0,1) with
x(0) = 0 and x(1) = 1 and using uniform h-refinements.

Gauss-Lobatto quadrature rule is used:
i.e. replace "return GaussJacobiQuadratureLineRule(ref_el, m)"
with "return GaussLobattoLegendreQuadratureLineRule(ref_el, m)"
in the file FIAT/quadrature.py

This code is part of
Mesh partial differential equations (PDEs): application and implementation,
written by Roisin Hill and Niall Madden
Contact: Roisin Hill
For: <cite paper>
"""

from fenics import *
import math

# Set degree for the Gauss-Lobatto quadrature rule
parameters["form_compiler"]["quadrature_degree"] =  4

# Problem parameters
epsilon = 1E-6 	 	# perturbation factor of the physical PDE
N = 32  			# mesh intervals in final mesh

# parameters for rho(x)
sigma = 2.5			# related to degree of elements
b = 0.99			# lower bound on reaction coefficient
q = 0.5				# proportion of mesh points in layer
K = q/(sigma*(1-q))	# proportion of points in the layer region

# Parameters for h-refinement steps
N_start = 4			# initial number of mesh intervals
N_step = N_start	# current number of mesh intervals
step_index = 0		# initial step number

# Define right side of PDE and initial solution
f = Expression('0.0', degree = 2)
x_0 = Expression('x[0]', degree = 2)

# Define stopping criteria tolerance and intiial value
residual_stop_TOL = 0.2
residual_stop = residual_stop_TOL +1

# Lists for mesh and function space names
mesh_list =  ["mesh0", "mesh1","mesh2","mesh3","mesh4", "mesh5", "mesh6", "mesh7", "mesh8", "mesh9"]
V_list = ["V0","V1","V2","V3","V4","V5","V6","V7","V8","V9"]

# Computactional function space parameters
def comp_space(N):
	meshc = UnitIntervalMesh(N)			# Uniform unit interval mesh
	V = FunctionSpace(meshc, 'P', 1)	# Function space V with linear elements
	v = TestFunction(V)					# Test function on V
	x = TrialFunction(V)				# Trial function on V
	return meshc, V, v, x

# Define weak form of MPDE
def weak_form(rho, xN, x, v, f, meshz, V):
	bc = DirichletBC(V, x_0, "on_boundary")	# Define Dirichlet boundary condition
	a = rho(xN)*inner(grad(x),grad(v))*dx(meshz)# Left side of weak_form
	L = f*v*dx				# Right side of weak_form
	xN = Function(V)		# Numerical solution in V
	solve(a==L, xN, bc)		# Solve, applying the boundary conditions strongly
	return xN

# Define rho(x) for MPDE
def rho(xN):
	return conditional((K*(b/epsilon)*exp(-b*xN/(sigma*epsilon)))>1.0, K*(b/epsilon)*exp(-b*xN/(sigma*epsilon)), 1.0)

# Calculate the norm of the residual to use as stopping criterion
def calc_residual_stop(rho, V, meshN):
	residual = assemble((rho(xN)*xN.dx(0)*v.dx(0)-f*v)*dx) # residual of xN
	bcr = DirichletBC(V, '0.0', "on_boundary")
	bcr.apply(residual)					# apply Dirichlet boundary conditions
	residual_func = Function(V)
	residual_func.vector()[:] = residual.get_local() # Residual function on V
	residual_stop = norm(residual_func, 'L2', meshN) # L2-norm of residual
	return residual_stop

# Initial function space parameters
mesh_list[0], V_list[0], v, x = comp_space(N_step) #

# Initial value for xN
xN = interpolate(x_0, V_list[0])

# Iterate through uniform h-refinements
while N_step < N:

	# Calculate solution twice on each mesh size
	for i in range(0,2):
		xN = weak_form(rho, xN, x, v, f, mesh_list[step_index], V_list[step_index])

	# Double the number of mesh intervals
	N_step = N_step*2
	step_index = round(math.log(N_step/N_start,2))

	# Generate computational function space on the finer mesh
	mesh_list[step_index], V_list[step_index], v, x = comp_space(N_step)

	# Interpolate the solution onto the new computational function space
	xN = interpolate(xN, V_list[step_index])

# Calculate solution and L2 norm of the residual on final function space
while residual_stop > residual_stop_TOL:

	xN = weak_form(rho, xN, x, v, f, mesh_list[step_index], V_list[step_index])
	residual_stop = calc_residual_stop(rho, V_list[step_index], mesh_list[step_index])

# Generate the physical mesh
meshp = UnitIntervalMesh(N)
meshp.coordinates()[:,0]  = xN.compute_vertex_values()[:]
