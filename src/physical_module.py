### IMPORTED LIBRARIES ########
#General libraries:
import numpy as np
import scipy as sp
import os
import time
from PIL import Image
from numpy import linalg as LA
import scipy.sparse as spsparse
import scipy.linalg as splin
#For property evaluation:
import cvxopt 
import cvxopt.cholmod




def GetHomProp2D_PlaneStress(MetaDesign,E1,nu1,E2,nu2,Amat=np.eye(2)):
# Get unit cell full stiffness matrix Kuc - assume plane Strain, thickness = 1
# 1 for stiff material;  0 for soft material (air)
    nelx = MetaDesign.shape[1]
    nely = MetaDesign.shape[0]
    ndof = 2*(nelx+1)*(nely+1)

    KA = np.array([[12.,  3., -6., -3., -6., -3.,  0.,  3.],
                   [ 3., 12.,  3.,  0., -3., -6., -3., -6.],
                   [-6.,  3., 12., -3.,  0., -3., -6.,  3.],
                   [-3.,  0., -3., 12.,  3., -6.,  3., -6.],
                   [-6., -3.,  0.,  3., 12.,  3., -6., -3.],
                   [-3., -6., -3., -6.,  3., 12.,  3.,  0.],
                   [ 0., -3., -6.,  3., -6.,  3., 12., -3.],
                   [ 3., -6.,  3., -6., -3.,  0., -3., 12.]])
    KB = np.array([[-4.,  3., -2.,  9.,  2., -3.,  4., -9.],
                   [ 3., -4., -9.,  4., -3.,  2.,  9., -2.],
                   [-2., -9., -4., -3.,  4.,  9.,  2.,  3.],
                   [ 9.,  4., -3., -4., -9., -2.,  3.,  2.],
                   [ 2., -3.,  4., -9., -4.,  3., -2.,  9.],
                   [-3.,  2.,  9., -2.,  3., -4., -9.,  4.],
                   [ 4.,  9.,  2.,  3., -2., -9., -4., -3.],
                   [-9., -2.,  3.,  2.,  9.,  4., -3., -4.]])
    
    KE1 = E1/(1-nu1**2)/24*(KA+nu1*KB);
    KE2 = E2/(1-nu2**2)/24*(KA+nu2*KB);

    # FE: Build the index vectors for the for coo matrix format.
    edofMat=np.zeros((nelx*nely,8),dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            el = ely+elx*nely
            n1=(nely+1)*elx+ely
            n2=(nely+1)*(elx+1)+ely
            edofMat[el,:]=np.array([2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3,2*n2, 2*n2+1, 2*n1, 2*n1+1])

    # Construct the index pointers for the coo format
    iK = np.kron(edofMat,np.ones((8,1))).flatten()
    jK = np.kron(edofMat,np.ones((1,8))).flatten()  
    sK=((KE1.flatten()[np.newaxis]).T * MetaDesign.flatten()).flatten('F') + ((KE2.flatten()[np.newaxis]).T * (1-MetaDesign).flatten()).flatten('F')
    Kuc = spsparse.coo_matrix((sK,(iK,jK)),shape=(ndof,ndof)).tocsr()
    
    # Get unit cell periodic topology
    M = np.eye((nelx+1)*(nely+1))
    M[0,[nely,(nely+1)*nelx,(nelx+1)*(nely+1)-1]] = 1
    M[1:nely,range(1+(nely+1)*nelx,nely+(nely+1)*nelx)] = np.eye(nely-1)
    M[np.arange((nely+1),(nely+1)*nelx,(nely+1)),np.arange(2*nely+1,(nely+1)*nelx,(nely+1))] = 1
    M = M[np.sum(M,axis=0)<2,:].T

    # Compute homogenized elasticity tensor
    B0 = spsparse.kron(M,np.eye(2));
    Bep = np.array([[Amat[0,0], 0., Amat[1,0]/2],
                    [0., Amat[1,0], Amat[0,0]/2],
                    [Amat[0,1], 0., Amat[1,1]/2],
                    [0., Amat[1,1], Amat[0,1]/2]])
    BaTop = np.zeros(((nelx+1)*(nely+1),2),dtype=np.single)
    BaTop[(nely+1)*nelx+np.arange(0,nely+1),0] = 1
    BaTop[np.arange(nely,(nely+1)*(nelx+1),(nely+1)),1] = -1;
    Ba = np.kron(BaTop,np.eye(2,dtype=float))
    
    TikReg = spsparse.eye(B0.shape[1])*1e-8
    F = (Kuc.dot(B0)).T.dot(Ba)
    Kg = (Kuc.dot(B0)).T.dot(B0)+TikReg   
    Kg = (0.5 * (Kg.T + Kg)).tocoo()
    Ksp = cvxopt.spmatrix(Kg.data,Kg.row.astype(int),Kg.col.astype(int))
    Fsp = cvxopt.matrix(F)
    cvxopt.cholmod.linsolve(Ksp,Fsp)
    D0 = np.array(Fsp)
    Da = -B0.dot(D0)+Ba
    Kda = (Kuc.dot(Da)).T.dot(Da);
    Chom = (Kda.dot(Bep)).T.dot(Bep) / LA.det(Amat);
    Modes = Da.dot(Bep)
    return Chom

def cleanimages(designs, csf=1):
    cleaned = []
    n = len(designs[:,0,0,0])
    for i in range(n):
        design = designs[i,0,:,:]
        design_res = np.array(Image.fromarray(design).resize((image_size // csf, image_size // csf), Image.BICUBIC))
        design_out = 1.0 - (design_res>0.5)
        cleaned.append(design_out)
    return np.array(cleaned)


def S_prime_max_angle(Shom):
    alpha_arange = np.linspace(0, np.pi, 60)
    S_list = []
    for alpha in alpha_arange:
        A_matrix = np.array([[np.cos(alpha), np.sin(alpha)],
                            [-np.sin(alpha), np.cos(alpha)]])   
        N_prime = np.array([[A_matrix[0,0]**2, A_matrix[0,1]**2, A_matrix[0,0]*A_matrix[0,1]],
                            [A_matrix[1,0]**2, A_matrix[1,1]**2, A_matrix[1,0]*A_matrix[1,1]],
                        [2*A_matrix[0,0]*A_matrix[1,0], 2*A_matrix[0,1]*A_matrix[1,1], A_matrix[0,0]*A_matrix[1,1]+A_matrix[0,1]*A_matrix[1,0] ]]) 
        Shom_prime = np.dot(N_prime, Shom).dot(N_prime.T)
        S_list.append(1/Shom_prime[0,0])
    argmax = np.argmax(S_list)
    angle = alpha_arange[argmax]
    angle = np.cos(angle*2)
    return np.array(([angle, np.max(S_list)/600]))

def Estimator(designs, image_size=64, csf=2):
    E1 = 2066.54
    nu1 = 0.337
    E2 = 0.498
    nu2 = 0.5
    Eeff = []
    cleaned = []
    designs = np.array(designs)
    main_angle = []
    n = len(designs[:,0,0,0])
    for i in range(n):
        design = designs[i,0,:,:]
        design_res = np.array(Image.fromarray(design).resize((image_size // csf, image_size // csf), Image.BICUBIC))
        design_out = 1.0 - (design_res>0.5)
        cleaned.append(design_out)
        

    for i in range(n):
        Ehom = GetHomProp2D_PlaneStress(cleaned[i].T,E1,nu1,E2,nu2)
        Shom = splin.inv(Ehom)
        Eeff.append(0.5*(1/Shom[0,0]+1/Shom[1,1]))
        main_angle.append(S_prime_max_angle(Shom))

    return np.array(main_angle)#np.array([Eeff, main_angle])