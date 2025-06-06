# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 11:52:42 2025

@author: edgar

Library of the TP3

"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npl
import matplotlib.tri as tri
import math
import numpy.random as rd

def generate_mesh_cartesian_domain(x_m,x_M,y_m,y_M,Nx,Ny,para,plot):
    '''
    Parameters
    ----------
    x_m,x_M,y_m,x_M : floats
        Corner of our cartesain domain
    Nx, Ny: int
        Number of points of each axis, the number of total 
        points will be : Nx * Ny.
        It is better if Nx=Ny
    para : 0 or 1 
        0 : regular mesh 
        1 : random mesh 
    plot : 0 or 1
        0: plot the mesh
        1: no plot of the mesh 
    Returns
    -------
    A triangle mesh of our domain.  

    '''
    
    if para==0 : 
        # Uniform meshgrid 
        x=np.linspace(x_m,x_M,Nx+2)
        y=np.linspace(y_m,y_M,Ny+2)

        X,Y=np.meshgrid(x,y)

        X=X.flatten()
        Y=Y.flatten()

        triang = tri.Triangulation(X, Y)
        if plot==1 : 
            # Representation of the mesh
            plt.figure(0)
            plt.gca().set_aspect('equal')
            plt.triplot(X,Y,triang.triangles, 'b-', lw=0.5)
            plt.title('Maillage')
            plt.show()
        
        
        return(triang)
    
    
    if para ==1: 
        Rx=rd.random([Nx+2,Nx+2])
        Rx[:,0]=0
        Rx[:,-1]=0
        Ry=rd.random([Ny+2,Ny+2])
        Ry[0,:]=0
        Ry[-1,:]=0
        x=np.linspace(x_m,x_M,Nx+2)#+4*((x_M-x_m)/(5*(Nx+2)))*Rx
        y=np.linspace(y_m,y_M,Ny+2)#+4*((y_M-y_m)/(5*(Ny+2)))*Ry
        
        X,Y=np.meshgrid(x,y)
        X=X+0.9*((x_M-x_m)/((Nx+2)))*Rx
        Y=Y+0.9*((y_M-y_m)/((Ny+2)))*Ry
        
        X=X.flatten()
        Y=Y.flatten()
        
        triang = tri.Triangulation(X, Y)
        if plot==1 : 
            # Representation of the mesh
            plt.figure(1)
            plt.gca().set_aspect('equal')
            plt.triplot(X,Y,triang.triangles, 'b-', lw=0.5)
            plt.title('maillage')
            plt.show()
        
        return(triang)

def shape_function_gradients(coords):
    '''
    Parameters
    ----------
    coords = [(x1, y1), (x2, y2), (x3, y3)]
        The 3 coordinates corner of a triangle
    Returns
    -------
    The 3 gradiens et the area of the triangle  

    '''
    # coords = [(x1, y1), (x2, y2), (x3, y3)]
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]

    # Area of the triangle
    area = 0.5 * np.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

    # The 3 gradients of the coordinates functions 
    grad_N1 = np.array([y2 - y3, x3 - x2]) / (2 * area)
    grad_N2 = np.array([y3 - y1, x1 - x3]) / (2 * area)
    grad_N3 = np.array([y1 - y2, x2 - x1]) / (2 * area)

    return grad_N1, grad_N2, grad_N3, area

def Plot_f(triang,u,i):
        '''
        Parameters
        ----------
        triang : matplotlib.tri._triangulation.Triangulation
            Contains of the information about the mesh, 
        u : np.array 
            Array to plot
        Returns
        -------
        None 

        '''
        plt.figure(i)
        plt.gca().set_aspect('equal')
        plt.tripcolor(triang.x,triang.y,triang.triangles,u, shading='flat')
        plt.colorbar()
        plt.title('Cas f=1')
        plt.show()
        return None 

def assembly_matrix(x_m,x_M,y_m,y_M,Nx,Ny,para,plot_mesh,M,R): 
    '''
    Parameters
    ----------
    x_m,x_M,y_m,x_M : floats
        Corner of our cartesain domain
    Nx, Ny: int
        Number of points of each axis, the number of total 
        points will be : Nx * Ny.
        It is better if Nx=Ny
    para : 0 or 1 
        0 : regular mesh 
        1 : random mesh 
    plot_mesh : 0 or 1
        0: plot the mesh 
        1: not plot the mesh 
    M : array numpy 
        Mass matrix
    R: array numpy
        Stifness matrix
    Returns
    -------
    M,R the mass et stifness matrix assembled

    '''
    triang=generate_mesh_cartesian_domain(x_m,x_M,y_m,y_M,Nx,Ny,para,plot_mesh)
    
    NTri=np.shape(triang.triangles)[0]
    NSom=np.shape(triang.x)[0]

    #Table with nodes coordinates
    TabSom=np.zeros([NSom,2])
    TabSom[:,0]=triang.x
    TabSom[:,1]=triang.y

    # Table with triangle nodes
    TabTri=triang.triangles
    # Initialize Global Element Matrices : Mass Matrix and Stiffness matrix
    
    # Initialize Local Element Matrices on each triangle
    M0 = np.diag(2*np.ones(3),0)+np.diag(np.ones(2),-1)+np.diag(np.ones(2),1)
    M0[2,0]=1
    M0[0,2]=1
    R0 = np.zeros((3,3))
    ##### Algorithm to implement the mass matrix and the stifness matrix
    for i in range(NTri):

        Tri=TabTri[i] 
        coords = TabSom[Tri]
        grad_N1, grad_N2, grad_N3, area = shape_function_gradients(coords)
        Tri=TabTri[i] 
        
        ############ Elementary matrix
        ##Mass matrix
        Me=(area/12)*M0 
        
        ##Stiffness matrix
        coords = TabSom[Tri] # coords = [(x1, y1)=Tab[0], (x2, y2)=Tab[0], (x3, y3)]
        grad_N1, grad_N2, grad_N3, area = shape_function_gradients(coords)
        R0 = np.array([
                [np.dot(grad_N1, grad_N1), np.dot(grad_N1, grad_N2), np.dot(grad_N1, grad_N3)],
                [np.dot(grad_N2, grad_N1), np.dot(grad_N2, grad_N2), np.dot(grad_N2, grad_N3)],
                [np.dot(grad_N3, grad_N1), np.dot(grad_N3, grad_N2), np.dot(grad_N3, grad_N3)]
        ]) * area

        #####Global assembly 
        for j in range(3):
            for l in range(3):
                
                R[Tri[j],Tri[l]]+=R0[j,l]
                M[Tri[j],Tri[l]]+=Me[j,l]
           
        
        
    return(M,R)
  
    
    