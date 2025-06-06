# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 12:51:24 2025

@author: edgar
"""
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npl
import matplotlib.tri as tri
import math
import TP3_Library as method


######Domain : a square of size 2
Nx=20
Ny=20

x_m=1.
x_M=3.
y_m=1.
y_M=3.

triang=method.generate_mesh_cartesian_domain(x_m, x_M, y_m, y_M, Nx, Ny, 1,0)
NTri=np.shape(triang.triangles)[0]
NSom=np.shape(triang.x)[0]

#Table with nodes coordinates
TabSom=np.zeros([NSom,2])
TabSom[:,0]=triang.x
TabSom[:,1]=triang.y

# Table with triangle nodes
TabTri=triang.triangles
M = np.zeros((NSom,NSom)) #Mass matrix
R = np.zeros((NSom,NSom)) #Stiffness matrix
M,R=method.assembly_matrix(x_m, x_M, y_m, y_M, Nx, Ny,1, 0,M,R)


#print(R,M)



##### Algorithm to implement the right hand side
### Cas f=1 
F1=np.zeros(NSom)
for i in range(NSom):
     F1[i]=np.sum(M[i,:])   

#Or 
#f=np.ones(NSom)
#F12=M@f

u_h1= npl.solve(R+M,F1)

# plt.figure(1)
# plt.gca().set_aspect('equal')
# plt.tripcolor(triang.x,triang.y,triang.triangles,u_h1, shading='flat')
# plt.colorbar()
# plt.title('Cas f=1')
# plt.show()

####### Test for one specific u 

#the result right hand side 
def f(x1,x2): 
    return((2*np.pi**2+1)* np.cos(np.pi*x1)*np.cos(np.pi*x2))

#the theorical solution
def u_sol(x1,x2):
    return(np.cos(np.pi*x1)*np.cos(np.pi*x2))

###Compute the right hand side with the mass matrix 
F_val=f(triang.x,triang.y)
F2=M@F_val 
#Resolution
u_h2= npl.solve(R+M,F2)
#Theorical interpolation
u_theo=u_sol(triang.x,triang.y)

#L2-error : ||I(u)-u_h||² = (I(u)-u)^T M (I(u)-u)
error_L2 = np.sqrt((u_theo - u_h2).T @ M @ (u_theo - u_h2))
norm_u_L2 = np.sqrt(np.sum((u_theo**2) * np.sum(M, axis=1)))
#print("Norme L^2 de la différence",error_L2)# Calcul de l'erreur en norme L2

# plt.figure(2)
# plt.gca().set_aspect('equal')
# plt.tripcolor(triang.x,triang.y,triang.triangles,u_theo, shading='flat')
# plt.colorbar()
# plt.title('Solution théorique')
# plt.show()


# plt.figure(3)
# plt.gca().set_aspect('equal')
# plt.tripcolor(triang.x,triang.y,triang.triangles, u_h2, shading='flat')
# plt.colorbar()
# plt.title('Solution approchee par EF P1')
# plt.show()


# plt.figure(4)
# plt.gca().set_aspect('equal')
# plt.tripcolor(triang.x,triang.y,triang.triangles, u_theo-u_h2, shading='flat')
# plt.colorbar()
# plt.title(r'$ I_h(u)-u_h$')
# plt.show()



###########Calcul of the mesh order 
'''
If the mesh is regular, it's easy to compute the mesh order as the max edge is the diagonal
of every triangle. 
However, in general, we need to compute all the 3 lenghts of all the edges of the triangle
'''

 
def edge_lenghts(TabTri,TabSom,i):
    '''
    Parameters
    ----------
    TabTri: array numpy
        Table with the triangle of our mesh 
    TabSom : array numpy
        Table with the coordinates of all the triangle of our mesh 
    i : int 
         Index in order to browse the tables
    -------
    Lenghts : np.array(3)
        The 3 lenghts of our triangle 
    '''
    
    Tri=TabTri[i] # our triangle
    coords=TabSom[Tri]
    x1,y1=coords[0] 
    x2,y2=coords[1]
    x3,y3=coords[2] 
    lengths = [np.sqrt((x2 - x1)**2 + (y2 - y1)**2),np.sqrt((x3 - x1)**2 + (y3 - y1)**2),np.sqrt((x3 - x2)**2 + (y3 - y2)**2)]
    
    return lengths

def Order_mesh(TabTri,TabSom):
       '''
       Parameters
       ----------
       TabTri: array numpy
           Table with the triangle of our mesh 
       TabSom : array numpy
           Table with the coordinates of all the triangle of our mesh 
    
       -------
       h : float
           The max lenght in our mesh.
       ''' 
       NTri=np.shape(TabTri)[0]  #Number of triangle
       Lenghts=np.array([])
       for i in range(NTri):
           Lenghts=np.concatenate((Lenghts,edge_lenghts(TabTri, TabSom, i)))
          
       h=np.max(Lenghts)  #Order of our mesh 
       return(h)
           
h=Order_mesh(TabTri, TabSom)

##Test : 
# print(h)
# # We can verify be calculate the diagonal of one random triangle:  

# print(max(edge_lenghts(TabTri, TabSom, 0)))
##Test ok
'''
The random mesh provide a bigger h, which seems logical
'''


######### Order simulation 
'''
Know, in order to verify that the convergence of P1 finite element are 1. 
We will simulate with different step, we just have to increase the number of points Nx and Ny
'''
L_N=[5,10,15,20,25,30,35,40,45,50,55,60]
L_h=[]
L_error_L2=[]
L_Tri=[]
for N in L_N: 
    #Parameter of our mesh 
    Nx=N
    Ny=N

    x_m=1.
    x_M=3.
    y_m=1.
    y_M=3.
    
    triang=method.generate_mesh_cartesian_domain(x_m, x_M, y_m, y_M, Nx, Ny, 1,0)   ##Changer ici pour avoir un mesh aléatoire
    NTri=np.shape(triang.triangles)[0]
    NSom=np.shape(triang.x)[0]

    #Table with nodes coordinates
    TabSom=np.zeros([NSom,2])
    TabSom[:,0]=triang.x
    TabSom[:,1]=triang.y

    # Table with triangle nodes
    TabTri=triang.triangles
    
    M = np.zeros((NSom,NSom)) #Mass matrix
    R = np.zeros((NSom,NSom)) #Stiffness matrix
    
    #Assembling the matrix
    M,R=method.assembly_matrix(x_m, x_M, y_m, y_M, Nx, Ny, 0,0,M,R)
    
    #Right hand side
    F_val=f(triang.x,triang.y)
    F2=M@F_val 
    
    
    #Solving
    u_h=npl.solve(R+M,F2)
    
    #Theorical interpolation
    u_theo=u_sol(triang.x,triang.y)
    
    
    #List of step
    L_h.append(Order_mesh(TabTri, TabSom))
    
    
    #L2-error : ||I(u)-u_h||² = (I(u)-u)^T M (I(u)-u)
    L_error_L2.append(np.sqrt((u_theo - u_h).T @ M @ (u_theo - u_h)))
    
    #Number of triangle
    L_Tri.append(NTri)


# Calcul de la pente de la décroissance (ordre de convergence)
log_h = np.log(L_h)
log_L2 = np.log(L_error_L2)
slope, intercept = np.polyfit(log_h, log_L2, 1)


plt.plot(L_Tri,L_h,label=r"$\mathbb{P}_1$ finite element")
plt.title("Step of the mesh in function of the number of triangle")
plt.ylabel("h : step of the mesh")
plt.xlabel("Number of triangle of the mesh")
plt.grid()
plt.legend()
plt.show()


plt.loglog(L_h, L_error_L2, marker='o', linestyle='-', label=r"$\mathbb{P}_1$ finite element")
# Tracer la pente de référence (décroissance d'ordre 1)
plt.loglog(L_h, np.exp(intercept) * L_h**slope, linestyle='--', label=f'Pente : {slope:.2f}')
plt.ylabel(r"$ \|\mathcal{I}_h(u)-u_h\|_{L^2}$")
plt.xlabel(" h ")
plt.title(r"Order of convergence of $\mathbb{P}_1$ finite element")
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()








