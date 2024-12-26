import torch
import numpy as np
from scipy.spatial.distance import squareform,cdist

from pynamod.energy.energy_constants import *


class Energy:
    def __init__(self,K_free=1,K_elec=1,K_bend=1):
        
        self.force_matrix = None
        self.average_step_params = None
        self.K_free = K_free
        self.K_elec = K_elec
        self.K_bend = K_bend
        self.restraints = []
    
    def set_energy_matrices(self,CG_structure):
        AVERAGE,FORCE_CONST,DISP = get_consts_olson_98()
        pairtypes = [pair.pair_name for pair in CG_structure.dna.pairs_list]
        self._set_matrix(pairtypes,'force_matrix',FORCE_CONST)
        self._set_matrix(pairtypes,'average_step_params',AVERAGE)
        self._set_real_space_force_mat(CG_structure) 
    
    def add_restraints(self,restraints):
        '''
        input: list of restraints
        ''' 
        self.restraints += restraints
        
    def move_to_cuda(self):
        self.force_matrix.to('cuda')
        self.average_step_params.to('cuda')
        self.radii_sum_prod.to('cuda')
        self.epsilon_mean_prod.to('cuda')
        self.charges_multipl_prod.to('cuda')
        self.restraints.to('cuda')
    
    def get_full_energy(self,all_coords,CG_structure):
        return self._get_elastic_energy(all_coords.local_params) + self._get_real_space_total_energy(all_coords) + self._get_restraint_energy(CG_structure)
        
        
    def _set_matrix(self,pairtypes,attr,ref):
        matrix = torch.zeros((len(pairtypes),*ref['CG'].shape),dtype=torch.double)
        for i in range(len(pairtypes)-1):
            step = str(pairtypes[i][0]+pairtypes[i+1][0])
            matrix[i] = torch.Tensor(ref[step])
        setattr(self,attr,matrix)
    
    
    def _set_real_space_force_mat(self,CG_structure,ignore_neighbors=5):
        radii = CG_structure.radii
        epsilons = CG_structure.eps
        charges = CG_structure.charges
        
        self._set_dist_mat_slice(charges.shape[0],ignore_neighbors=ignore_neighbors)
        
        self.radii_sum_prod = self._triform(torch.Tensor(np.add.outer(radii,radii)))
        self.epsilon_mean_prod = self._triform(torch.Tensor(np.multiply.outer(epsilons,epsilons)))/2
        self.charges_multipl_prod = self._triform(torch.Tensor(np.multiply.outer(charges,charges)))
        
    
    def _set_dist_mat_slice(self,length,ignore_neighbors):
        sl = torch.arange(length**2).reshape(length,length)
        sl = torch.triu(sl, diagonal=ignore_neighbors).reshape(-1)
        self.dist_mat_slice = sl[sl>0]
        
    def _triform(self,square_mat):
        return square_mat.reshape(-1)[self.dist_mat_slice]

        
    def _get_elastic_energy(self,steps_params):
        params_dif = steps_params - self.average_step_params
        dif_matrix = torch.matmul(params_dif.reshape(-1,6,1), params_dif.reshape(-1,1,6))
        return self.K_bend*torch.einsum('ijk,ijk',dif_matrix, self.force_matrix)/2.0
    
    
    def _get_real_space_total_energy(self,all_coords,free_energy_func='sigmoid'):
        '''
        Supported free energy functions: sigmoid, Lennard Jones (LD) or function(dist_matrix,radii_sum_prod,epsilon_mean_prod)
        '''
        origins = all_coords.origins.reshape(-1,3)
        dist_matrix = self._triform(torch.cdist(origins,origins,compute_mode= 'donot_use_mm_for_euclid_dist'))
        electrostatic_e = self.charges_multipl_prod/dist_matrix
        if free_energy_func == 'sigmoid':
            free_energy = self.radii_sum_prod/(torch.sqrt(1/self.epsilon_mean_prod+dist_matrix**2))
        elif free_energy_func == 'LD':
            free_energy = self.epsilon_mean_prod*((self.radii_sum_prod/dist_matrix)**12-(self.radii_sum_prod/dist_matrix)**6)
        elif callable(free_energy_func):
            free_energy = free_energy_func(dist_matrix,self.radii_sum_prod,self.epsilon_mean_prod)
        else:
            raise ValueError('unknown function name')
        return self.K_elec*torch.sum(torch.nan_to_num(electrostatic_e, posinf=0, neginf=0)) + self.K_free*torch.sum(torch.nan_to_num(free_energy, posinf=0, neginf=0))
    

    def _get_restraint_energy(self,CG_structure):
        if self.restraints:
            return sum([restraint.get_restraint_energy(CG_structure) for restraint in self.restraints])
        else:
            return 0
    