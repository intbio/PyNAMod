import numpy as np
import torch
from scipy.spatial.distance import squareform,cdist

from pynamod.energy_constants import *


class Energy:
    def __init__(self):
        self.AVERAGE,self.FORCE_CONST,self.DISP = get_consts_olson_98()
        self.force_matrix = None
        self.average_step_params = None
        
    
    def set_matrix(self,pairtypes,movable_steps,attr,ref):
        matrix = np.zeros((len(pairtypes),*ref['CG'].shape))
        for i in range(len(pairtypes)-1):
            step = str(pairtypes[i][0]+pairtypes[i+1][0])
            matrix[i] = ref[step]
        setattr(self,attr,matrix[movable_steps])
    

    
    def set_real_space_force_mat(self,DNA_structure,movable_steps):
        radii = np.hstack([np.array(DNA_structure.radii)[movable_steps]]+DNA_structure.get_proteins_attr('cg_radii'))
        epsilons = np.hstack([np.array(DNA_structure.eps)[movable_steps]]+DNA_structure.get_proteins_attr('eps'))
        charges = np.hstack([np.array(DNA_structure.charges)[movable_steps]]+DNA_structure.get_proteins_attr('cg_charges'))
        self.radii_sum_prod = squareform(np.add.outer(radii,radii),checks=False)
        self.epsilon_mean_prod = squareform(np.multiply.outer(epsilons,epsilons),checks=False)/2
        self.charges_multipl_prod = squareform(np.multiply.outer(charges,charges),checks=False)
        
    
    def set_energy_params(self,K_free,K_elec):
        self.K_free = K_free
        self.K_elec = K_elec
    
    def set_energy_matrices(self,DNA_structure,movable_steps):
        pairtypes = [pair.pair_name for pair in DNA_structure.pairs_list]
        self.set_matrix(pairtypes,movable_steps,'force_matrix',self.FORCE_CONST)
        self.set_matrix(pairtypes,movable_steps,'average_step_params',self.AVERAGE)
        self.set_real_space_force_mat(DNA_structure,movable_steps)

        
    def get_bend_energy(self,steps_params,movable_steps):
        params_dif = steps_params[movable_steps] - self.average_step_params
        dif_matrix=np.matmul(params_dif[:, :, np.newaxis], params_dif[:, np.newaxis, :])
        return np.einsum('ijk,ijk',dif_matrix, self.force_matrix)/2.0
    
    
    def calc_real_space_total_energy(self,DNA_structure,movable_steps,free_energy_func='sigmoid'):
        '''
        Supported free energy functions: sigmoid, Lennard Jones (LD) or function(dist_matrix,radii_sum_prod,epsilon_mean_prod)
        '''
        all_coords = np.vstack([DNA_structure.base_ref_frames[movable_steps,3,:3]]+[protein.cg_beads_pos for protein in DNA_structure.proteins])
        dist_matrix = squareform(cdist(all_coords,all_coords),checks=False)
        electrostatic_e = self.K_elec*np.sum(self.charges_multipl_prod/dist_matrix)
        if free_energy_func == 'sigmoid':
            free_energy = self.K_free*np.sum(self.radii_sum_prod/(np.sqrt(1/self.epsilon_mean_prod+dist_matrix**2)))
        elif free_energy_func == 'LD':
            free_energy = self.K_free*np.sum(self.radii_sum_prod/(np.sqrt(1/self.epsilon_mean_prod+dist_matrix**2)))
        elif callable(free_energy_func):
            free_energy = self.K_free*free_energy_func(dist_matrix,self.radii_sum_prod,self.epsilon_mean_prod)
        else:
            raise ValueError('unknown function name')
        
        return electrostatic_e + free_energy
    
    def get_full_energy(self,DNA_structure,movable_steps):
        return self.get_bend_energy(DNA_structure.steps_params,movable_steps) + self.calc_real_space_total_energy(DNA_structure,movable_steps)
    
    
    
    