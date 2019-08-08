import numpy as np
from numba import jit
from pynamod.geometry_transformations import *

def get_params_for_single_step_debug(o1,o2,R1,R2):
    z1=R1[:,2]
    z2=R2[:,2]

    print( "hinge axis")
    hinge= np.cross(z1,z2)/np.linalg.norm(np.cross(z1,z2))
    print (hinge)

    print( "Roll Tilt angle, degrees")
    RollTilt= (np.arccos(vec_dot_product(z1,z2)))
    print(np.rad2deg(RollTilt))

    print ("R_hinge")
    R_hinge=rmat(hinge,-0.5*RollTilt)
    print (R_hinge)

    print ("R2'")
    R2p= np.dot(R_hinge,R2)
    print (R2p)

    print ("R1'")
    R1p=np.dot(rmat(hinge,0.5*RollTilt),R1)
    print (R1p)

    print ("Rm")
    Rm=(R1p+R2p)/2.0
    Rm[:,0]/=np.linalg.norm(Rm[:,0])
    Rm[:,1]/=np.linalg.norm(Rm[:,1])
    Rm[:,2]/=np.linalg.norm(Rm[:,2])
    print (Rm)

    print ("om")
    om=(o1+o2)/2.0

    print (om)

    print ("Shift Slide Rise")
    [shift,slide,rise]=np.dot((o2-o1),Rm)
    print ([shift,slide,rise])
    #print(np.cross(R1p[:,1],R2p[:,1]))
    twist= np.rad2deg(np.arccos(vec_dot_product(R1p[:,0],R2p[:,0])))
    phi=np.arccos(vec_dot_product(hinge,Rm[:,1]))
    print('phi ',np.rad2deg(phi))
    roll=RollTilt*np.cos(phi)
    tilt=RollTilt*np.sin(phi)
    return(shift,slide,rise,np.rad2deg(tilt),np.rad2deg(roll),twist)

def get_params_for_single_step_stock(o1,o2,R1,R2):
    z1=R1[:,2]
    z2=R2[:,2]
    hinge= np.cross(z1,z2)/np.linalg.norm(np.cross(z1,z2))
    RollTilt= (np.arccos(vec_dot_product(z1,z2)))
    R_hinge=rmat(hinge,-0.5*RollTilt)
    R2p= np.dot(R_hinge,R2)
    R1p=np.dot(rmat(hinge,0.5*RollTilt),R1)
    Rm=(R1p+R2p)/2.0
    om=(o1+o2)/2.0
    [shift,slide,rise]=np.dot((o2-o1),Rm)
    twist= np.rad2deg(np.dot(np.cross(R1p[:,1],R2p[:,1]),Rm[:,2]))
    phi=np.dot(np.cross(hinge,Rm[:,1]),Rm[:,2])

    roll=RollTilt*np.cos(phi)
    tilt=RollTilt*np.sin(phi)
    return shift,slide,rise,np.rad2deg(tilt),np.rad2deg(roll),twist

@jit(nopython=True)
def get_params_for_single_step_numba(o1,o2,R1,R2):
    z1=R1[:,2]
    z2=R2[:,2]
    hinge= cross(z1,z2)/np.linalg.norm(cross(z1,z2))
    RollTilt= (np.arccos(vec_dot_product(z1,z2)))
    R_hinge=rmat(hinge,-0.5*RollTilt)
    R2p= np.dot(R_hinge,R2)
    
    R1p=np.dot(rmat(hinge,0.5*RollTilt),R1)
    Rm=(R1p+R2p)/2.0
    om=(o1+o2)/2.0
    (shift,slide,rise)=np.dot((o2-o1),Rm)
    twist= vec_dot_product(cross(R1p[:,1],R2p[:,1]),Rm[:,2])
    phi=vec_dot_product(cross(hinge,Rm[:,1]),Rm[:,2])

    roll=RollTilt*np.cos(phi)
    tilt=RollTilt*np.sin(phi)
    return shift,slide,rise,np.rad2deg(tilt),np.rad2deg(roll),np.rad2deg(twist)

@jit(nopython=True)
def get_params_for_single_step_numba2(o1,o2,R1,R2):
    z1=R1[:,2]
    z2=R2[:,2]
    v_length=length(cross_product(z1,z2))
    hinge= [i/v_length for i in cross_product(z1,z2)]
    RollTilt= (np.arccos(np.dot(z1,z2)))
    
    R_hinge=rmat_c(hinge,-0.5*RollTilt)
    R2p= np.dot(R_hinge,R2)
    R1p=np.dot(rmat_c(hinge,0.5*RollTilt),R1)
    Rm=(R1p+R2p)/2.0
    om=(o1+o2)/2.0
    [shift,slide,rise]=np.dot((o2-o1),Rm)
    twist=  np.rad2deg(np.dot(np.array(cross_product(R1p[:,1],R2p[:,1])),Rm[:,2]))
    phi=np.dot(np.array(cross_product(hinge,Rm[:,1])),Rm[:,2])

    roll=RollTilt*np.cos(phi)
    tilt=RollTilt*np.sin(phi)
    return shift,slide,rise,np.rad2deg(tilt),np.rad2deg(roll),twist


######################################################################################
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
    
    gamma=sqrt(tilt**2 + roll**2)

    #norm=length(np.array([tilt,roll,0]))
    roll_tilt_axis=np.array([tilt,roll,0])
    roll_tilt_axis/=length(roll_tilt_axis)

    phi=calc_phi(roll_tilt_axis)
    
    rmat_inmat(z,phi,temp1)
    rmat_inmat(y,gamma/2,temp2)
    temp3=np.dot(temp2,temp1)
    rmat_inmat(z,twist/2-phi,temp1)   
    
    rm=np.dot(temp1, temp3)
    
    rmat_inmat(z,twist/2+phi,temp1)
    rmat_inmat(y,gamma,temp2)
    temp3=np.dot(temp2,temp1)
    rmat_inmat(z,twist/2-phi,temp1)     
    
    r2=np.dot(temp1, temp3)
    
    o2=np.dot(np.array([shift,slide,rise]),rm.T)

    R2_exp = np.dot(R1_exp,r2)
    o2_exp=o1_exp+np.dot(o2,R1_exp.T)
    return(o2_exp,R2_exp)

@jit
def get_ori_and_mat_from_step(shift,slide,rise,tilt,roll,twist,R1_exp,o1_exp):
    tilt=np.deg2rad(tilt)
    roll=np.deg2rad(roll)
    twist=np.deg2rad(twist)

    #x=np.array([1.,0.,0.])
    y=np.array([0.,1.,0.])
    z=np.array([0.,0.,1.])
    temp1=np.zeros((3,3))
    temp2=np.zeros((3,3))
    
    gamma=sqrt(tilt**2 + roll**2)

    norm=np.linalg.norm(np.array([tilt,roll,0]))
    roll_tilt_axis=[tilt/norm,roll/norm,0]

    phi=calc_phi(np.array(roll_tilt_axis))
    temp1=rmat(z,twist/2-phi)
    temp2=rmat(y,gamma/2)
    
    rm=np.dot(rmat(z,twist/2-phi), np.dot(rmat(y,gamma/2), rmat(z,phi)))
    r2=np.dot(rmat(z,twist/2-phi), np.dot(rmat(y,gamma), rmat(z,twist/2+phi)))
    o2=np.dot(np.array([shift,slide,rise]),rm.T)

    R2_exp = np.dot(R1_exp,r2)
    o2_exp=o1_exp+np.dot(o2,R1_exp.T)
    return(o2_exp,R2_exp)

@jit
def rebuild_by_full_par_frame_numba(full_par_frame):
    R1_exp=np.identity(3)
    o1_exp=np.array([0.0,0.0,0.0])
    bp_frames=np.zeros((full_par_frame.shape[0],4,4))
    bp_frames[0,:3,:3]=R1_exp
    bp_frames[0,3,:3]=o1_exp
    for i in range(1,full_par_frame.shape[0]):
        params=full_par_frame[i,6:12]
        bp_frames[i,3,:3],bp_frames[i,:3,:3]=get_ori_and_mat_from_step_opt(params[0],params[1],params[2],
                                                                           params[3],params[4],params[5],
                                                                           bp_frames[i-1,:3,:3],bp_frames[i-1,3,:3])

    return(bp_frames)
#bp_frames=rebuild_by_full_par_frame(full_par_frame)

def rebuild_by_full_par_frame(full_par_frame):
    R1_exp=np.identity(3)
    o1_exp=np.array([0.0,0.0,0.0])
    bp_frames=np.zeros((full_par_frame.shape[0],4,4))
    bp_frames[0,:3,:3]=R1_exp
    bp_frames[0,3,:3]=o1_exp
    for i,params in enumerate(full_par_frame[1:,6:12],1):
        o1_exp,R1_exp=get_ori_and_mat_from_step_opt(*params,R1_exp,o1_exp)
        bp_frames[i,:3,:3]=R1_exp
        bp_frames[i,3,:3]=o1_exp
    return(bp_frames)