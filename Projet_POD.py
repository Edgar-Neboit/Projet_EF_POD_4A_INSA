#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Affinely parametrized linear BVP:
     - div( lambda(mu) * grad(u) ) + w * grad(u) = f  in domain
                                       u = g  on bdry dirichlet
                         - lambda(mu) nabla(u).n = 0 on bdry Neumann
with w: given velocity field

Double input parameters: mu1, mu2
    
Goal: Solve this BVP by an offline-online strategy based on a POD.
 
'''

from dolfin import *
from fenics import *
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eigh
import time
import random
import numpy.linalg as npl
import scipy
import scipy.linalg   
import math
from mpl_toolkits.mplot3d import axes3d

# The PDE parameter: diffusivity lambda(mu)
def Lambda(mu):#here mu=[mu1,mu2] 
    mu1=mu[0]
    return np.exp(mu1-11)

# Function to compute the RB dimension (= Nrb)
def energy_number(epsilon_POD,lam):
    # lam: eigenvalues table
    # return the eignvalue number corresponding to energy_ratio
    index_min = 0; s = 0.;s1=np.sum(lam)
    for i in range(len(lam)):
        if s < s1*(1-epsilon_POD):
            s += lam[i]
            index_min = index_min + 1
    return index_min

# Dirichlet boundary conditions
tol_bc = 1.e-10
def u_bdry_0(x, on_boundary):
    return bool(on_boundary and (near(x[0], 0, tol_bc)))
def u_bdry_1(x, on_boundary):
    return bool(on_boundary and (near(x[0], 1, tol_bc)))

###################################################
#    Offline phase
###################################################

# Physical and numerical parameters
# Mesh and function spaces
NP =  35; print('Number of mesh points NP = ', NP)  # 35 for comp
mesh = UnitSquareMesh(NP,NP)
k = 1 ; print('Order of the Lagrange FE k = ', k)  # EF order 1
V = FunctionSpace(mesh, "CG", int(k))
V_vec = VectorFunctionSpace(mesh, "CG", int(k))
N_h = V.dim(); print('Resulting number of nodes N_h = ', N_h)
coordinates = mesh.coordinates()
# Trial and test function
u, v = TrialFunction(V), TestFunction(V)

# Snapshots number
print('How many snapshots do I compute ? ')
N_s = int(input())

# The parameter range mu
# The input parameter mu_0
mu0 = 0.7
# The input parameter mu
mu_min = 1.0; mu_max = 10. # range of values 
print('Range values for mu1: [',mu_min,',',mu_max,']')
mu1 = np.linspace(mu_min,mu_max,N_s)
mu2 = np.linspace(0,np.pi/2,N_s)
mu=[[mu1[i],mu2[i]] for i in range(N_s)]

### Save for the NN
np.save('Parameters_two_params',mu)

print("Set of parameters:",mu)
# Plot of the parameter space
Param =  np.zeros(len(mu))
for i in range(len(mu)):
    Param[i] = Lambda(mu[i])
print("Param=",Param)
fig = plt.figure()
ax = fig.gca() 
ax.scatter(mu1, Param) 
plt.title("The parameters space")
ax.set_xlabel('The physical parameter mu1')
ax.set_ylabel('Lambda(mu1)')
plt.legend()
plt.show()

# Plot the grid of parameters: mu1 vs mu2
fig2 = plt.figure()
ax2 = fig2.gca()
ax2.plot(mu1, mu2, marker='o', linestyle='-', color='red')
plt.title("Grid of parameters: $\\mu_1$ vs $\\mu_2$")
ax2.set_xlabel('The physical parameter $\\mu_1$')
ax2.set_ylabel('The physical parameter $\\mu_2$')
plt.grid(True)
plt.show()

L_max1=Lambda([np.max(mu[0]),0])
L_max2=Lambda([np.max(mu[1]),0])

L_min1=Lambda([np.min(mu[0]),0])
L_min2=Lambda([np.min(mu[1]),0])

mu_max1,mu_max2=np.max(mu1),np.max(mu2)
mu_min1,mu_min2=np.min(mu1),np.min(mu2)
print("mu_max1=",mu_max1,"mu_max2=",mu_max2)
print("mu_min1=",mu_min1,"mu_min2=",mu_min2)
# RHS of the PDE model (parametrized by mu2)
mu2 = 1.0                     # Example of value for mu2
A = 10.0
L = 2.0

# f(x) = A * cos(mu2 * L * x[0])
# Define the expression with only the degree specified
f_expr = Expression("A * cos(mu2 * L * x[0])", element=V.ufl_element(), A=A, mu2=mu2, L=L)
f = interpolate(f_expr, V)

# Velocity field
vel_amp = 1e+2; print('vel_amp =',vel_amp)
vel_exp = Expression(('(1.+abs(cos(2*pi*x[0])))', 'sin(2*pi/0.2*x[0])'), element = V.ufl_element())
#vel_exp = Expression(('0.', '0.'), element = V.ufl_element())
vel = vel_amp * interpolate(vel_exp,V_vec)

print('#')
print('# Computation of the M snapshots')
print('#')
S = np.zeros((N_h,N_s)) # Snaphots matrix
uh = np.zeros(N_h)
t_0 =  time.time()
for m in range(N_s):
    print('snapshot #',m,' : mu = ',mu[m])
    diffus = Lambda(mu[m])
    print('snapshot #',m,' : Lambda(mu) = ',diffus)
    # Variational formulation
    F = diffus * dot(grad(v),grad(u)) * dx + v * dot(vel, grad(u)) * dx - f * v * dx
    # Stabilization of the advection term by SUPG 
    r = - diffus * div( grad(u) ) + dot(vel, grad(u)) - f #residual
    vnorm = sqrt( dot(vel, vel) )
    h = MaxCellEdgeLength(mesh)
    delta = h / (2.0*vnorm)
    F += delta * dot(vel, grad(v)) * r * dx
    # Create bilinear and linear forms
    a = lhs(F); L = rhs(F)
    # Dirichlet boundary conditions
    u_diri0_exp = Expression('1.', degree=u.ufl_element().degree())
    bc0 = DirichletBC(V, u_diri0_exp, u_bdry_0)
    # Solve the problem
    u_mu = Function(V)
    solve(a == L, u_mu,bc0)
    # Buid up the snapshots matrices U 
    uh = u_mu.vector().get_local()[:]
    S[:,m] = uh # dim.  N_h * N_s

# Plot of the manifold in 3D
# On sélectionne 3 degrés de liberté (indices arbitraires ou bien espacés)
i1, i2, i3 = 0, N_h//2, N_h-1 
# on fixe ici 3 composantes sur lequels on va tracer nos solutions en ces 3 points
# ces 3 indices correspondents à un noyeu du maillage le point (i1,i2,i3), on trace alors : 
# pour tout \mu \in P, un vecteur (u(\mu)^{i1},u(\mu)^{i2},u(\mu)^{i3})
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(S[i1, :], S[i2, :], S[i3, :], 'o-', label=r"$(u(\mu)^{i}, u(\mu)^{j},u(\mu)^{k})$")
ax.set_xlabel(f"i=DoF {i1}")
ax.set_ylabel(f"j=DoF {i2}")
ax.set_zlabel(f"k=DoF {i3}")
ax.set_title("Snapshot manifold in 3d")
ax.legend()
plt.show()

##Save for the NN
np.save('Snapshots_two_params',S)

# Assemble of the rigidity matrix (FE problem in V_h)
f_exp1 = Expression('0.0', element = V.ufl_element())
f1 = interpolate(f_exp1,V)
u1, v1 = TrialFunction(V), TestFunction(V)
F1 =  dot(grad(v1),grad(u1)) * dx +  u1 * v1 * dx + f1 * v1 * dx
a1 = lhs(F1); L1 = rhs(F1)
# Assemble & get the matrix N_h x N_h
A_ass1, F1 = assemble_system(a1, L1)
# For L2 norm we have:
A_N_h1 = np.identity(N_h)

####################  POD  method ###################
# Computation of the correlation matrix for L2 norm
# Computation of the left singular vector from the eigenvectors of C
print("#")
print("# POD computation")
print("#")

# 1. Matrix of correlation (L2 norm, thus scalar product)
C = np.dot(S.T, np.dot(A_N_h1, S))  # A_N_h1 is id 

# 2. Eigen values and correlation matrix
lam, W = eigh(C)  
lam = lam[::-1]      # decrease order
W = W[:, ::-1]

# 3. Eigen modes
Phi = np.zeros((N_h, N_s))
for i in range(N_s):
    if lam[i] > 1e-14:
        Phi[:, i] = (1. / np.sqrt(lam[i])) * np.dot(S, W[:, i])

# 4. Normalisation
for i in range(N_s):
    norm_phi = np.sqrt(np.dot(Phi[:, i], Phi[:, i]))
    if norm_phi > 1e-14:
        Phi[:, i] /= norm_phi

# 5. plots
plt.figure()
plt.semilogy(np.arange(1, N_s + 1), lam, 'o-')
plt.title("Eigen values of the correlation matrix")
plt.xlabel("Indice")
plt.ylabel("Eigen values (log scale)")
plt.grid(True)
plt.show()


# Plot of the eigenvalues
eigen_val=lam
decay = np.arange(len(eigen_val))
fig = plt.figure()
ax = fig.gca() 
ax.plot(decay, abs(eigen_val), label='Eigenvalues',color='r') 
plt.title("The decay of the eigen values")
ax.set_xlabel('The eigenvalues index')
ax.set_ylabel('The eigen values')
#plt.xscale("log")
plt.yscale("log")
plt.legend()
# Plot in bar chart
width = 0.5
p =plt.bar(decay,eigen_val, width, color='b')
plt.title('The M eigenvalues');plt.ylabel('Eigenvalues')
plt.show()

# Tolerance epsilon to determine the number of modes Nrb
print('Give a tolerance to compute Nrb')
epsilon_POD = float(input())

# Computation of the number of modes Nrb
eigen_val_normalized=eigen_val/npl.norm(eigen_val)
Nrb = energy_number(epsilon_POD,eigen_val_normalized)  #function to compute Nrb see course
print('This corresponds to Nrb = ',Nrb)

# Define a range of tolerances
tolerances = np.linspace(1e-9, 1e-2, 10)
Nrb_values = [energy_number(tol, eigen_val_normalized) for tol in tolerances]

# Plot
plt.figure()
plt.plot(tolerances, Nrb_values, 'o-')
plt.title(r"Reduced dimension $N_{rb}$ vs tolerance $\varepsilon_{POD}$")
plt.xlabel(r"Tolerance $\varepsilon_{POD}$")
plt.ylabel(r"Reduced dimension $N_{rb}$")
plt.xscale('log')
plt.grid(True)
plt.show()

# Truncation of the Reduced Basis 
t_1 =  time.time()
tcpu1=t_1
trunc_error = np.sum(eigen_val[Nrb:])  # POD truncation error (theoretical)
print("POD theoretical truncation error:", trunc_error)

# The error estimation satisfied by the POD method
err=0
U, Sigma, Vt = np.linalg.svd(S, full_matrices=False)
Vrb = U[:, :Nrb]    # trunc
projection = Vrb @ Vrb.T @ S    #Projection in L2 , its the Brb matrix 
real_proj_error = np.linalg.norm(S - projection, 'fro')**2  #projection min problem 
print("Real projection error:", real_proj_error)

##Save for the NN
np.save('Brb_two_params',projection)

err=trunc_error - real_proj_error
print("Diff 2 errors =",err)
##################################################
#         Online phase
##################################################
print('#'); print('# Online phase begins... #')
#
## New parameter value mu (must be within the same intervall [mu_min,mu_max])

# New parameter value mu (must be within the same interval [mu_min, mu_max])

print(f'Choose a new value of the parameter mu1 (within the same interval [{mu_min1}, {mu_max1}])')
mu1 = float(input())

print(f'Choose a new value of the parameter mu2 (within the same interval [{mu_min2}, {mu_max2}])')
mu2 = float(input())


# Diffusivity parameter
mu=[mu1,mu2]
diffus = Lambda(mu)

print('   You will get the RB solution for mu = ',mu)
print('   This corresponds to lambda(mu) = ',diffus)

# Assemble the rigidity matrix...
# Variational formulation
u, v = TrialFunction(V), TestFunction(V)
F = diffus * dot(grad(v),grad(u)) * dx + v * dot(vel, grad(u)) * dx - f * v * dx
# SUPG stabilisation
r = - diffus * div( grad(u) ) + dot(vel, grad(u)) - f # Residual
vnorm = sqrt( dot(vel, vel) )
h = MaxCellEdgeLength(mesh); delta = h / (2.0*vnorm)
F += delta * dot(vel, grad(v)) * r * dx
# Create bilinear and linear forms
a = lhs(F); L = rhs(F)
u_diri0_exp = Expression('1.', degree=u.ufl_element().degree())
bc0 = DirichletBC(V, u_diri0_exp, u_bdry_0)
# Assemble and get the matrix N_h x N_h plus the RHS vector N_h
A_ass, F = assemble_system(a, L, bc0)
A_N_h = A_ass.array()
F_N_h=F.get_local()
# Stiffness matrix & RHS of the reduced system
# The reduced stiffness matrix: Brb^T A_N_h Brb

A_rb=Vrb.T@A_N_h@Vrb

# The reduced RHS
F_rb=Vrb.T@F_N_h

# Solve the reduced system
u_rb = np.linalg.solve(A_rb, F_rb)

tcpu2=time.time()
print('RB solution CPU-time = ',tcpu2- tcpu1)

#
# Difference ("error") between the HR FE solution and the RB solution
#
# The RB solution in the complete FE basis: Urb = B_rb^T . urb 
Urb=Vrb@u_rb

# Transform the RB solution to a Fenics object
Urb_V = Function(V)
Urb_V.vector().set_local(Urb)
#Save the Reduced solution for comp
np.save("Uh_POD",Urb)


#
# Computation of the current HR FE solution 
print('Compute the complete FE solution to be compared with uRB !')
# By following the usual way, it would give:
#uh = Function(V)
#solve(a == L, uh,bc0)
#Uh = uh.vector().get_local() # npy vector 
# Compute the HR FE solution by solving the FE system A_N_h . Uh = F
# This enables to compare the CPU time
tcpu3 = time.time()
Uh = np.linalg.solve(A_N_h,F)
#Save the finite element solution for comp
np.save("Uh_FE",Uh)
tcpu4 = time.time()
print('FE solution CPU-time = ',tcpu4 - tcpu3)
# The relative diff. vector & its FEniCS object
error_vec = abs(Uh-Urb)/abs(Uh)
error_func = Function(V); error_func.vector().set_local(error_vec)

# Plot of the FE, RB and relative error functions 
# FE solution
uh = Function(V)
uh.vector().set_local(Uh)
p=plot(uh, title="The FE solution")#,mode='color',vmin=1.0, vmax=1.0000040)
p.set_cmap("rainbow"); plt.colorbar(p); plt.show()

# RB solution
p=plot(Urb_V, title="The Reduced Basis solution")#,mode='color',vmin=1.0, vmax=1.0000040)
p.set_cmap("rainbow"); plt.colorbar(p); plt.show()

# Relative difference solution
error_func = Function(V)
error_func.vector().set_local(error_vec)
p=plot(error_func,title="The relative diff ",mode='color')#,vmin=0.0, vmax=0.024)
p.set_cmap("rainbow"); plt.colorbar(p); plt.show()

# Computation of the relative errors in L2, 2-norm and norm max 
# Relative diff. ("error") in norm max
error_norm_max = np.linalg.norm(Uh-Urb,ord=np.inf)
# Norm max of HR FE solution
norm_max_Uh = np.linalg.norm(Uh,ord=np.inf)
# Relative error in norm max
error_relative_norm_max = (error_norm_max)/(norm_max_Uh)
print('Relative diff. in norm max = ',error_relative_norm_max)

# Relative diff. in norm 2
error_norm2 = np.linalg.norm(Uh-Urb)
# Norm 2 of HR FE solution
norm2_Uh = np.linalg.norm(Uh)
# Relative error in norm 2
error_relative_norm2 = (error_norm2)/(norm2_Uh)
print('Relative diff. in norm 2 = ',error_relative_norm2)

#########################Save solution for comp
np.save('Uh_POD',Urb)



# Relative diff. in L2 norm first method using the energy norm
# Error in L2 norm or in L2 if A_N_h = I_N_h
error_L2 = np.sqrt(np.dot((Uh-Urb),np.dot(A_N_h1,Uh-Urb)))
# H1 norm of HR FE solution
L2_Uh = np.sqrt(np.dot(Uh,np.dot(A_N_h1,Uh)))
# Relative error in H1 norm
error_relative_L2_norm = error_L2/L2_Uh
print("Relative diff H1 norm=",error_relative_L2_norm)

# Relative diff. in L2 norm second method (Fenics norm)
# Function to stor the diff between HR FE and RB solutions
diff = Function(V)
# Get the corresponding vectors
diff.vector().set_local(Uh-Urb)
# Error in H1 norm using FEniCS
error_L2_fenics = norm(diff, 'L2', mesh)
# H1 norm of HR FE solution using FEniCS
L2_Uh_fenics = norm(uh, 'L2', mesh)
print('#')
print('#')
print('#')

discarded_eigenvalues = np.sum(eigen_val[Nrb:])
s_eigen=np.sum(eigen_val)
relative_error_pod = discarded_eigenvalues / s_eigen
# Print out performances and errors of the POD method
print('POD method performance summary:')
print('-----------------------------------')
print('Offline phase CPU time        :', t_1 - t_0, 'seconds')
print('Full-order (FE) solve time    :', tcpu4 - tcpu3, 'seconds')
print('Reduced-order (RB) solve time :', tcpu2 - tcpu1, 'seconds')
print('Number of reduced basis (Nrb) :', Nrb)
print('-----------------------------------')
print('Relative error (max norm)     :', error_relative_norm_max)
print('Relative error (L2 norm)      :', error_relative_norm2)
print('Relative error (L2 integral)  :', error_relative_L2_norm)
print('FEniCS relative L2 error norm :', error_L2_fenics / L2_Uh_fenics)
print('-----------------------------------')
print('Discarded eigenvalues sum     :', discarded_eigenvalues)
print('Total eigenvalues sum         :', s_eigen)
print('POD error estimate (method 1) :', relative_error_pod )
print('-----------------------------------')



