import numpy as np

def get_rotation_and_offset_ref_frame_to_obj(ref_mat,ref_ori,obj_mat,obj_ori):
    rotation=np.dot(np.linalg.inv(ref_mat.T),obj_mat)
    ofset=np.dot((obj_ori-ref_ori),np.linalg.inv(ref_mat.T))
    return(rotation,ofset)

def get_obj_orientation_and_location(ref_mat,ref_ori,rotation,offset):
    return(np.dot(ref_mat.T, rotation),np.dot(offset,ref_mat.T)+ref_ori)