import numpy as np

#AVERAGE,FORCE_CONST,DISP=get_consts_olson_98()
from pynamod.utils import get_movable_steps
from scipy.spatial.distance import pdist, cdist,squareform
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

def get_real_space_force_mat(dna_beads,ncp_beads,misc_beads=None,
                             dna_r=5,ncp_r=60,misc_r=5,
                             dna_eps=0.5,ncp_eps=1,misc_eps=0,
                             dna_q=0,ncp_q=0,misc_q=0,dist_pairs=None,**kwargs):
    results={}
    dna_r=np.ones(dna_beads.shape[0])*dna_r if isinstance(dna_r,(int, float)) else dna_r
    ncp_r=np.ones(ncp_beads.shape[0])*ncp_r if isinstance(ncp_r,(int, float)) else ncp_r

    dna_eps=np.ones(dna_beads.shape[0])*dna_eps if isinstance(dna_eps,(int, float)) else dna_eps
    ncp_eps=np.ones(ncp_beads.shape[0])*ncp_eps if isinstance(ncp_eps,(int, float)) else ncp_eps

    dna_q=np.ones(dna_beads.shape[0])*dna_q if isinstance(dna_q,(int, float)) else dna_q
    ncp_q=np.ones(ncp_beads.shape[0])*ncp_q if isinstance(ncp_q,(int, float)) else ncp_q
        
    if not (misc_beads is None):
        misc_r=np.ones(misc_beads.shape[0])*misc_r if isinstance(misc_r,(int, float)) else misc_r
        misc_eps=np.ones(misc_beads.shape[0])*misc_eps if isinstance(misc_eps,(int, float)) else misc_eps
        misc_q=np.ones(misc_beads.shape[0])*misc_q if isinstance(misc_q,(int, float)) else misc_q
        radii=np.concatenate((dna_r,ncp_r,misc_r))
        epsilons=np.concatenate((dna_eps,ncp_eps,misc_eps))
        charges=np.concatenate((dna_q,ncp_q,misc_q))
    else:
        radii=np.concatenate((dna_r,ncp_r))
        epsilons=np.concatenate((dna_eps,ncp_eps))
        charges=np.concatenate((dna_q,ncp_q))
    
    results['radii_sum_prod']=squareform(np.add.outer(radii,radii),checks=False)
    results['epsilon_mean_prod']=squareform(np.add.outer(epsilons,epsilons),checks=False)/2
    results['charges_multipl_prod']=squareform(np.multiply.outer(charges,charges),checks=False)
    
    
    if not(dist_pairs is None):
        dist_mat=np.zeros((len(radii),len(radii)))
        index_mat=np.zeros((len(radii),len(radii)))
        pair_indexes=[]
        for pair in dist_pairs:
            indexes=[]
            for mate in ['a','b']:
                if pair[mate][0]=='ncp':
                    if pair[mate][1]<ncp_beads.shape[0]:
                        indexes.append(dna_beads.shape[0]+pair[mate][1])
                    else:
                        break
                elif pair[mate][0]=='misc':
                    if misc_beads is None:
                        break
                    if pair[mate][1]<misc_beads.shape[0]:
                        indexes.append(dna_beads.shape[0]+ncp_beads.shape[0]+pair[mate][1])
            if len(indexes)==2:
                index_mat[indexes[0],indexes[1]]=1
                dist_mat[indexes[0],indexes[1]]=pair['dist']
            pair_indexes=np.argwhere(squareform(index_mat,checks=False)).flatten()
            results['pair_indexes']=pair_indexes
            results['pair_distances']=squareform(dist_mat,checks=False)[pair_indexes]

    return(results)       
        

def calc_real_space_total_energy(all_coords,radii_sum_prod=None,epsilon_mean_prod=None,
                      charges_multipl_prod=None,
                      pair_indexes=None,pair_distances=None,K_excl=1,K_elec=1,K_dist=1,**kwargs):
    dist_matrix=pdist(all_coords)
    excluded_e_logi=0
    if not (radii_sum_prod is None):
        excluded_e_logi=K_excl*np.sum(epsilon_mean_prod*(1-1/(1+np.exp(-dist_matrix+radii_sum_prod))))
    #excluded_e_lj=np.sum(epsilon_mean_prod*((radii_sum_prod/dist_matrix)**12 - 2*(radii_sum_prod/dist_matrix)**6))
    electrostatic_e=0
    if not (charges_multipl_prod is None):
        electrostatic_e=K_elec*np.sum(charges_multipl_prod/dist_matrix)
    pair_dist_e=0
    if not(pair_indexes is None):
        pair_dist_e=K_dist*np.sum((dist_matrix[pair_indexes]-pair_distances)**2)
    return(excluded_e_logi+electrostatic_e+pair_dist_e,{'vdv':excluded_e_logi,'e':electrostatic_e,'restr':pair_dist_e})
