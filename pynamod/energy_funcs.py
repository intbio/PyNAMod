import numpy as np

#AVERAGE,FORCE_CONST,DISP=get_consts_olson_98()
from pynamod.utils import get_movable_steps

def get_force_matrix(pairtypes,movable_steps,FORCE_CONST):
    '''
    Constructs b.p. bending force matrix for the given sequnce and constants
    pairtypes - list of pairtypes e.g.  ['A-T', 'T-A', 'C-G',...,'A-T']
    movable_steps - list with b.p. step indexes which should be changed during optimization
                    can be obtained with utils.get_movable_steps()
    FORCE_CONST   - dict with b.p. bending force constants (see energy constants.get_consts_olson_98)
    '''
    force_matrix=[]
    for i in range(len(pairtypes)-1):
        step=str(pairtypes[i][0]+pairtypes[i+1][0])
        force_matrix.append(FORCE_CONST[step])
    force_matrix=np.array(force_matrix).astype(float)
    return(force_matrix[movable_steps])

def get_average_bpstep_frame(pairtypes,movable_steps,AVERAGE):
    '''
    Constructs average b.p. step frame for the given sequnce and constants
    pairtypes - list of pairtypes e.g.  ['A-T', 'T-A', 'C-G',...,'A-T']
    movable_steps - list with b.p. step indexes which should be changed during optimization
                    can be obtained with utils.get_movable_steps()
    AVERAGE       - dict with average b.p. step values (see energy constants.get_consts_olson_98)
    '''
    average_pbstep_frame=[]
    for i in range(len(pairtypes)-1):
        step=str(pairtypes[i][0]+pairtypes[i+1][0])
        average_pbstep_frame.append(AVERAGE[step])
    average_pbstep_frame=np.array(average_pbstep_frame).astype(float)
    return(average_pbstep_frame[movable_steps])

def get_bpstep_frame(full_par_frame,movable_steps):
    '''
    returns only the part involving b.p.step parameters (last 6 columns) from N x 12 full_par_frame
    '''
    return(full_par_frame[movable_steps+1,6:12])

def _calc_bend_energy(pbstep_frame,force_pbstep_matrix,average_pbstep_frame):
    '''
    Calculates bending energy for given pbstep_frame and force matrix
    pbstep_frame          - the part involving b.p.step parameters (last 6 columns) from N x 12 full_par_frame
    force_pbstep_matrix   - bending force matrix for the given sequnce and constants
                            see energy_funcs.get_force_matrix
    average_pbstep_frame  - average b.p. step frame for the given sequnce and constants
                            see energy_funcs.get_average_bpstep_frame
    returns the bending energy of dna conformation
    
    '''

    dif_frame=pbstep_frame.reshape(-1,6)-average_pbstep_frame
    #this is equivalent to dif_matrix=np.asarray([np.outer(row, row) for row in dif_frame])  but way faster
    dif_matrix=np.matmul(dif_frame[:, :, np.newaxis], dif_frame[:, np.newaxis, :])   
    #this is equivalent to np.sum(np.multiply(dif_matrix,force_pbstep_matrix)) but faster
    result = np.einsum('ijk,ijk',dif_matrix, force_pbstep_matrix)/2.0

    return(result)

def get_bend_energy(full_par_frame,pairs,movable_steps,FORCE_CONST,AVERAGE):
    '''
    Calculates bending energy for given full_par_frame and force constants
    full_par_frame - numpy array with N x 12 b.p. par frame
    pairs - list of pairtypes e.g.  ['A-T', 'T-A', 'C-G',...,'A-T']
    movable_steps - list with b.p. step indexes which should be changed during optimization
                    can be obtained with utils.get_movable_steps()
    FORCE_CONST   - dict with b.p. bending force constants (see energy constants.get_consts_olson_98)
    AVERAGE       - dict with average b.p. step values (see energy constants.get_consts_olson_98)
    
    '''
    force_matrix=get_force_matrix(pairs,movable_steps,FORCE_CONST)
    average_bpstep_frame=get_force_matrix(pairs,movable_steps,AVERAGE)
    bpstep_frame=get_bpstep_frame(full_par_frame,movable_steps)
    return(_calc_bend_energy(bpstep_frame,force_matrix,average_bpstep_frame))
