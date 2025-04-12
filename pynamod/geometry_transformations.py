import numpy as np
from numba import jit

from math import atan2
from math import pi

@jit
def rmat(axis, phi):
    axis=axis/ np.linalg.norm(axis)
    u1=axis[0]
    u2=axis[1]
    u3=axis[2]
    c=np.cos(phi)
    s=np.sin(phi)
    return np.array([[   c+(1-c)*u1*u1, (1-c)*u1*u2-u3*s, (1-c)*u1*u3+u2*s],
                     [(1-c)*u1*u2+u3*s,  c+(1-c)*u2*u2,(1-c)*u2*u3-u1*s],
                     [(1-c)*u1*u3-u2*s, (1-c)*u2*u3+u1*s,   c+(1-c)*u3*u3]])

@jit
def rmat_inmat(axis, phi,mat):
    axis=axis/ length(axis)
    u1=axis[0]
    u2=axis[1]
    u3=axis[2]
    c=np.cos(phi)
    s=np.sin(phi)
    mat[0,0]=c+(1-c)*u1*u1
    mat[0,1]=(1-c)*u1*u2-u3*s
    mat[0,2]=(1-c)*u1*u3+u2*s
    mat[1,0]=(1-c)*u1*u2+u3*s
    mat[1,1]=c+(1-c)*u2*u2
    mat[1,2]=(1-c)*u2*u3-u1*s
    mat[2,0]=(1-c)*u1*u3-u2*s
    mat[2,1]=(1-c)*u2*u3+u1*s
    mat[2,2]=c+(1-c)*u3*u3

@jit
def length(v):
    return np.sqrt(v[0]**2+v[1]**2+v[2]**2)
@jit
def vec_dot_product(v,w):
    return v[0]*w[0]+v[1]*w[1]+v[2]*w[2]

@jit
def cross(a, b):
    c = np.array((a[1]*b[2] - a[2]*b[1],
         a[2]*b[0] - a[0]*b[2],
         a[0]*b[1] - a[1]*b[0]))
    return c
@jit(nopython=True)
def cross_product(u,v):  
    dim = len(u)
    s = []
    for i in range(dim):
        if i == 0:
            j,k = 1,2
            s.append(u[j]*v[k] - u[k]*v[j])
        elif i == 1:
            j,k = 2,0
            s.append(u[j]*v[k] - u[k]*v[j])
        else:
            j,k = 0,1
            s.append(u[j]*v[k] - u[k]*v[j])
    return s
@jit
def determinant(v,w):
    return v[0]*w[1]-v[1]*w[0]
@jit
def inner_angle(v,w):
    cosx=vec_dot_product(v,w)/(length(v)*length(w))
    rad=np.arccos(cosx) # in radians
    return rad 
def calc_phi(A):
    cosx=vec_dot_product(A,[0.,1.,0.])/length(A)
    rad=np.arccos(cosx)
    #inner=inner_angle(A,[0.,1.,0.])
    ### phi_sign = vec_dot_product(cross_product(A,[0.,1.,0.]),[0.,0.,1])
    phi_sign = A[0]
    if phi_sign < 0:
        return -rad
    else:
        return rad
@jit(nopython=True)
def rmat_c(axis, phi):
    v_length=length(axis)
    hinge= [i/v_length for i in axis]
    u1=axis[0]
    u2=axis[1]
    u3=axis[2]
    c=np.cos(phi)
    s=np.sin(phi)
    return np.array([[   c+(1-c)*u1*u1, (1-c)*u1*u2-u3*s, (1-c)*u1*u3+u2*s],
                     [(1-c)*u1*u2+u3*s,  (c+(1-c)*u2*u2),(1-c)*u2*u3-u1*s],
                     [(1-c)*u1*u3-u2*s, (1-c)*u2*u3+u1*s,   c+(1-c)*u3*u3]])