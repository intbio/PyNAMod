import torch
import numpy as np
from scipy.spatial.distance import squareform,cdist

from pynamod.energy.energy_constants import *


class Energy:
    def __init__(self,K_free=1,K_elec=1,K_bend=1):
        
        self.force_matrix = None
        self.average_step_params = None
        self.K_free = torch.Tensor(K_free)
        self.K_elec = torch.Tensor(K_elec)
        self.K_bend = torch.Tensor(K_bend)
        self.restraints = []
    
    def set_energy_matrices(self,CG_structure,ignore_neighbors=5):
        AVERAGE,FORCE_CONST,DISP = get_consts_olson_98()
        pairtypes = [pair.pair_name for pair in CG_structure.dna.pairs_list]
        self._set_dist_mat_slice(CG_structure,ignore_neighbors=ignore_neighbors)
        self._set_matrix(pairtypes,'force_matrix',FORCE_CONST)
        self._set_matrix(pairtypes,'average_step_params',AVERAGE)
        self._set_real_space_force_mat(CG_structure)
        
    
    def add_restraints(self,restraints):
        '''
        input: list of restraints
        ''' 
        self.restraints += restraints
        
    def to(self,device):
        self.force_matrix = self.force_matrix.to(device)
        self.average_step_params = self.average_step_params.to(device)
        self.radii_sum_prod = self.radii_sum_prod.to(device)
        self.epsilon_mean_prod = self.epsilon_mean_prod.to(device)
        self.charges_multipl_prod = self.charges_multipl_prod.to(device)
        self.dist_mat_slice = self.dist_mat_slice.to(device)
        self.K_free = self.K_free.to(device)
        self.K_elec = self.K_elec.to(device)
        self.K_bend = self.K_bend.to(device)
    
    def get_full_energy(self,all_coords,CG_structure):
        return self._get_elastic_energy(all_coords.local_params) + self._get_real_space_total_energy(all_coords) + self._get_restraint_energy(CG_structure)
        
        
    def _set_matrix(self,pairtypes,attr,ref):
        matrix = torch.zeros((len(pairtypes),*ref['CG'].shape),dtype=torch.double)
        for i in range(len(pairtypes)-1):
            step = str(pairtypes[i][0]+pairtypes[i+1][0])
            matrix[i] = torch.Tensor(ref[step])
        setattr(self,attr,matrix)
    
    
    def _set_real_space_force_mat(self,CG_structure):
        radii = CG_structure.radii
        epsilons = CG_structure.eps
        charges = CG_structure.charges
        
        self.radii_sum_prod = self._triform(radii+radii.reshape(-1,1))
        self.epsilon_mean_prod = self._triform(torch.outer(epsilons,epsilons))/2
        self.charges_multipl_prod = self._triform(torch.outer(charges,charges))
        
    
    def _set_dist_mat_slice(self,CG_structure,ignore_neighbors):
        length = CG_structure.radii.shape[0]
        dna_length = CG_structure.dna.radii.shape[0]
        self.dist_mat_slice = torch.zeros(length,length,dtype=bool)
        dna_sl = torch.ones(dna_length,dna_length,dtype=bool)
        self.dist_mat_slice[:dna_length,:dna_length] = torch.triu(dna_sl, diagonal=ignore_neighbors)
        start = dna_length
        for protein in CG_structure.proteins:
            stop = protein.n_cg_beads+start
            self.dist_mat_slice[0:start,start:stop] = True
            start = stop
        
    def _triform(self,square_mat):
        return square_mat[self.dist_mat_slice]
    

        
    def _get_elastic_energy(self,steps_params):
        params_dif = steps_params - self.average_step_params
        dif_matrix = torch.matmul(params_dif.reshape(-1,6,1), params_dif.reshape(-1,1,6))
        return self.K_bend*torch.einsum('ijk,ijk',dif_matrix, self.force_matrix)/2.0
    

    
    def _get_real_space_total_energy(self,all_coords,free_energy_func='sigmoid'):
        '''
        Supported free energy functions: sigmoid, Lennard Jones (LD) or function(dist_matrix,radii_sum_prod,epsilon_mean_prod)
        '''
        origins = all_coords.origins.reshape(-1,3)
        return self._jit_get_real_e(origins,self.dist_mat_slice,self.radii_sum_prod,self.epsilon_mean_prod,self.charges_multipl_prod,self.K_elec,self.K_free)
    
    
    

    def _get_restraint_energy(self,CG_structure):
        if self.restraints:
            return sum([restraint.get_restraint_energy(CG_structure) for restraint in self.restraints])
        else:
            return 0
        

    def _jit_cdist(self,o):
        n = o.size(0)

        x = o.unsqueeze(1).expand(n, n, 3)
        y = o.unsqueeze(0).expand(n, n, 3)
        mat = torch.pow(x - y, 2).sum(2).pow(0.5)

        return mat


    def _jit_get_real_e(self,origins,dist_mat_slice,radii_sum_prod,epsilon_mean_prod,charges_multipl_prod,K_elec,K_free):

        n = origins.size(0)

        x = origins.unsqueeze(1).expand(n, n, 3)
        y = origins.unsqueeze(0).expand(n, n, 3)
        dist_matrix2 = torch.pow(x - y, 2).sum(2)

        dist_matrix2 = dist_matrix2[dist_mat_slice]
        electrostatic_e = charges_multipl_prod/dist_matrix2.pow(0.5)
        free_energy = radii_sum_prod/((1/epsilon_mean_prod+dist_matrix2).pow(0.5))
        torch.cuda.synchronize()
        return K_elec*electrostatic_e.sum() + K_free*free_energy.sum()