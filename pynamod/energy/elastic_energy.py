import torch
from pynamod.energy.energy_constants import *

class _Elastic_Energy_Calculator:
    def __init__(self,K_bend):
        self.K_bend = K_bend
        self.force_matrix = None
        self.average_step_params = None
        
    def set_energy_matrices(self,CG_structure):
        '''Creates matrices for energy calculation.
        
            Attributes:
            
            **CG_structure** - structure for which matrices are set.
            
            **ignore_neighbors** - number of neigboring dna pairs (in both sides) interactions with which are ignored in real space. Deafult 5.
            '''
        AVERAGE,FORCE_CONST,DISP = get_consts_olson_98()
        pairtypes = [pair.pair_name for pair in CG_structure.dna.pairs_list]
        self._set_matrix(pairtypes,'force_matrix',FORCE_CONST)
        self._set_matrix(pairtypes,'average_step_params',AVERAGE)
        
    def to(self,device):
        self.force_matrix = self.force_matrix.to(device)
        self.average_step_params = self.average_step_params.to(device)
        
    def get_energy(self,steps_params):
        params_dif = steps_params - self.average_step_params
        dif_matrix = torch.matmul(params_dif.reshape(-1,6,1), params_dif.reshape(-1,1,6))
        return self.K_bend*torch.einsum('ijk,ijk',dif_matrix, self.force_matrix)/2.0
        
    
    def _set_matrix(self,pairtypes,attr,ref):
        matrix = torch.zeros((len(pairtypes),*ref['CG'].shape),dtype=torch.double)
        for i in range(len(pairtypes)-1):
            step = str(pairtypes[i][0]+pairtypes[i+1][0])
            matrix[i+1] = torch.tensor(ref[step])
        setattr(self,attr,matrix)