import numpy as np
from numba import jit
from pynamod.geometry_transformations import *


def get_params_for_single_step_stock(o1,o2,R1,R2,pair_params=False):

    if pair_params:
        if vec_dot_product(R1[:,2],R2[:,2])<0:
            R1 = R1.copy()
            R1[:,1:] *= -1
            
    z1=R1[:,2]
    z2=R2[:,2]
    hinge= np.cross(z1,z2)/np.linalg.norm(np.cross(z1,z2))
    RollTilt= (np.arccos(vec_dot_product(z1,z2)))
    R_hinge=rmat(hinge,-0.5*RollTilt)
    R2p= np.dot(R_hinge,R2)
    
    R1p=np.dot(rmat(hinge,0.5*RollTilt),R1)
    
    Rm=(R1p+R2p)/2.0
    vectors_norm = np.linalg.norm(Rm,axis=0)
    Rm /= vectors_norm
    
    om=(o1+o2)/2.0
    [shift,slide,rise]=np.dot((o2-o1),Rm)
    
    twist= np.arccos(vec_dot_product(R1p[:,0],R2p[:,0]))
    twist_sign = vec_dot_product(cross_product(R1p[:,1],R2p[:,1]),Rm[:,2])
    if twist_sign < 0:
        twist *= -1
    
    phi_cos = vec_dot_product(hinge,Rm[:,1])
    if phi_cos > 1:
        phi = 0 
    else:
        phi = np.arccos(phi_cos)
    phi_sign = vec_dot_product(cross_product(hinge,Rm[:,1]),Rm[:,2])
    if phi_sign < 0:
        phi *= -1

    roll=RollTilt*np.cos(phi)
    tilt=RollTilt*np.sin(phi)
    return (shift,slide,rise,np.rad2deg(tilt),np.rad2deg(roll),np.rad2deg(twist)),om,Rm

@jit
def get_params_for_single_step(o1,o2,R1,R2,pair_params=False):
    
    if pair_params:
        if vec_dot_product(R1[:,2],R2[:,2])<0:
            R1 = R1.copy()
            #R1[:,1:] *= -1
            
    z1 = R1[:,2]
    z2 = R2[:,2]
    hinge = np.cross(z1,z2)/length(cross(z1,z2))
    RollTilt = (np.arccos(vec_dot_product(z1,z2)))
    R_hinge = rmat(hinge,-0.5*RollTilt)
    R2p = np.dot(R_hinge,R2)
    
    R1p=np.dot(rmat(hinge,0.5*RollTilt),R1)
    Rm=(R1p+R2p)/2.0
    vectors_norm = length(Rm)
    Rm /= vectors_norm
    om=(o1+o2)/2.0
    (shift,slide,rise)=np.dot((o2-o1),Rm)
    twist= np.arccos(vec_dot_product(R1p[:,0],R2p[:,0]))
    twist_sign = vec_dot_product(cross_product(R1p[:,1],R2p[:,1]),Rm[:,2])
    if twist_sign < 0:
        twist *= -1
    
    phi_cos = vec_dot_product(hinge,Rm[:,1])
    if phi_cos > 1:
        phi = 0 
    else:
        phi = np.arccos(phi_cos)
    phi_sign = vec_dot_product(cross_product(hinge,Rm[:,1]),Rm[:,2])
    if phi_sign < 0:
        phi *= -1

    roll=RollTilt*np.cos(phi)
    tilt=RollTilt*np.sin(phi)
    return (shift,slide,rise,np.rad2deg(tilt),np.rad2deg(roll),np.rad2deg(twist)),om,Rm


@jit
def get_ori_and_mat_from_step_opt(shift,slide,rise,tilt,roll,twist,R1_exp,o1_exp):
    tilt=np.deg2rad(tilt)
    roll=np.deg2rad(roll)
    twist=np.deg2rad(twist)

    #x=np.array([1.,0.,0.])
    y=np.array([0.,1.,0.])
    z=np.array([0.,0.,1.])
    temp1=np.zeros((3,3))
    temp2=np.zeros((3,3))
    
    gamma=np.sqrt(tilt**2 + roll**2)

    #norm=length(np.array([tilt,roll,0]))
    roll_tilt_axis=np.array([tilt,roll,0])
    roll_tilt_axis/=length(roll_tilt_axis)
    phi=calc_phi(roll_tilt_axis)
    rmat_inmat(z,twist/2-phi,temp1)
    rmat_inmat(y,gamma/2,temp2)
    temp3=np.dot(temp1,temp2)
    rmat_inmat(z,phi,temp2)   
    
    rm=np.dot(temp3, temp2)
    
    rmat_inmat(y,gamma,temp2)
    temp3=np.dot(temp1,temp2)
    rmat_inmat(z,twist/2+phi,temp2)  
    
    r2=np.dot(temp3, temp2)
    
    o2=np.dot(np.array([shift,slide,rise]),rm.T)

    R2_exp = np.dot(R1_exp,r2)
    o2_exp=o1_exp+np.dot(o2,R1_exp.T)
    return(o2_exp,R2_exp)



@jit
def rebuild_by_full_par_frame_numba(full_par_frame,start_bp_frame=None):
    
    length=full_par_frame.shape[0]
    bp_frames = np.zeros((length,4,4))
    
    if start_bp_frame is None:
        R1_exp = np.identity(3)
        o1_exp = np.array([0.0,0.0,0.0])
        bp_frames[0,:3,:3] = R1_exp
        bp_frames[0,3,:3] = o1_exp
    else:
        bp_frames[0] = start_bp_frame
        
    for i in range(1,length):
        params=full_par_frame[i]
        res = get_ori_and_mat_from_step_opt(params[0],params[1],params[2],params[3],params[4],params[5],bp_frames[i-1,:3,:3],bp_frames[i-1,3,:3])
        
        bp_frames[i,3,:3] = res[0]
        bp_frames[i,:3,:3] = res[1]

    return(bp_frames)


