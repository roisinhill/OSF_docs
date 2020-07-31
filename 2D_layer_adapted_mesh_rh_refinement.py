## 2D_layer_adapted_mesh_rh_refinement.py
"""
Generate a non-tensor product layer adapted mesh, for a singularly
perturbed PDE whose solution has layers that vary spatially near x = 1 and y = 1,
and a corner layer near (x,y) = (1,1)
by solving the MPDE -grad(M(x) grad(x(xi1, xi2))) = 0 for (xi1,xi2) in (0,1)^2 with
x(0,xi2) = (0,x_xi2=0),
x(1,xi2) = (1,x_xi2=0),
x(xi1,0) = (x_xi1=0,0),
x(xi1,1) = (x_xi1=0,1),
and using uniform h-refinements.

This code is part of
Mesh partial differential equations (PDEs): application and implementation,
written by Roisin Hill and Niall Madden
Contact: Roisin Hill
For: <cite paper>
"""

from fenics import *
import math

# Problem parameters
epsilon = 1E-2 			# perturbation factor of the physical PDE
N = 32  				# N+1 mesh points in each direction in final mesh

# Parameters for h refinement steps
N_start = 4				# initial mesh size
N_step = N_start		# current mesh size
step_index = 0			# initial step index

# Parameters for M(x)
sigma = 2.5				# dependant on method, often degree of elements +1 or +1.5
b = 0.99				# minimum coefficient of reaction or convectio term as appropriate
q = 0.5					# proportion of mesh points in layer
K = q/(sigma*(1-q))		# indicates the proportion of points in the layer region

# define right side of MPDE
f = Expression(('0.0','0.0'), degree = 2)

# Lists for mesh and function space names
mesh_list =  ["mesh0", "mesh1","mesh2","mesh3","mesh4", "mesh5", "mesh6", "mesh7", "mesh8", "mesh9"]
V_list = ["V0","V1","V2","V3","V4","V5","V6","V7","V8","V9"]

# generate computational function space
def comp_space(N):
	mesh = UnitSquareMesh(N, N, diagonal = "crossed") # uniform mesh on (0,1)^2
	V = VectorFunctionSpace(mesh, 'P', 1)	# Function space V with linear elements
	v = TrialFunction(V)					# Test function on V
	x = TestFunction(V)						# Trial function on V
	return mesh, V, v, x

# Define boundaries
TOL = 1E-14
def left_boundary( x, on_boundary):
	return  abs(x[0]) < TOL
def right_boundary(x, on_boundary):
	return abs(1-x[0]) < TOL
def bottom_boundary( x, on_boundary):
	return abs(x[1]) < TOL
def top_boundary( x, on_boundary):
	return abs(1-x[1]) < TOL

# Define Dirichlet boundary conditions
def boundary_conditions(V):
	bcl = DirichletBC(V.sub(0), '0.0', left_boundary)
	bcr = DirichletBC(V.sub(0), '1.0', right_boundary)
	bcb = DirichletBC(V.sub(1), '0.0', bottom_boundary)
	bct = DirichletBC(V.sub(1), '1.0', top_boundary)
	bcs = [bcl, bcr, bcb, bct]
	return bcs

# Define the matrix M(x) for MPDE
def M(xN):
	epsilon_y = Expression('epsilon*(1+xi1)*(1+xi1)', epsilon = epsilon, xi1 = xN.sub(0), degree = 2)
	epsilon_x = Expression('epsilon*(2-xi2)*(2-xi2)', epsilon = epsilon, xi2 = xN.sub(1), degree = 2)
	M = Expression((('K*(b/epsilon_x)*exp(-b*(1-xi1)/(sigma*epsilon_x))>1 ? K*(b/epsilon_x)*exp(-b*(1-xi1)/(sigma*epsilon_x)) : 1','0'),\
	('0','K*(b/epsilon_y)*exp(-b*(1-xi2)/(sigma*epsilon_y))>1 ? K*(b/epsilon_y)*exp(-b*(1-xi2)/(sigma*epsilon_y)) :1')),\
	K=K, b=b,sigma=sigma, epsilon_x = epsilon_x, epsilon_y = epsilon_y, xi1 = xN.sub(0), xi2 = xN.sub(1), degree = 4)
	return M

# Define the weak form
def weak_form(M, xN, x, v, f, bcs, meshz, V):
	a = inner(M(xN)*grad(x), grad(v))*dx(meshz) # Left side of weak weak_form
	L = dot(f,v)*dx								# Right side of weak weak_form
	xN = Function(V)							# Numerical solution in V
	solve(a==L, xN, bcs)						# Solve, applying the DirichletBC strongly
	return xN

# Initial function space parameters
mesh_list[0], V_list[0], v, x = comp_space(N_step)

# Set  intial value for xN: xN(xi1, xi2) = (xi1, xi2)
a = inner(grad(x),grad(v))*dx
L = dot(f,v)*dx
xN = Function(V_list[0])
solve(a==L, xN, boundary_conditions(V_list[0]))

# Iterate through uniform h-refinements
while N_step < N:

	# Generate solution four times on each mesh size
	for i in range (0,4):
		xN = weak_form(M, xN, x, v, f, boundary_conditions(V_list[step_index]), mesh_list[step_index], V_list[step_index])

	# Double the number of mesh intervals in each direction
	N_step = N_step*2
	step_index = round(math.log(N_step/N_start,2))

	# Generate computational function space on the finer mesh
	mesh_list[step_index], V_list[step_index], v, x = comp_space(N_step)

	# Interpolate solution onto the new computational function space
	xN = interpolate(xN,V_list[step_index])

# Calculate solution in final function space
iterations = 0
while iterations < 5:

	xN = weak_form(M, xN, x, v, f, boundary_conditions(V_list[step_index]), mesh_list[step_index], V_list[step_index])
	iterations +=1

# Generate the physical mesh
xiX, xiY = xN.split(True)
meshp = UnitSquareMesh(N,N)
meshp.coordinates()[:,0] = xiX.compute_vertex_values()[0:(N+1)*(N+1)]
meshp.coordinates()[:,1] = xiY.compute_vertex_values()[0:(N+1)*(N+1)]
