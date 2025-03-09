import torch
from pynamod.geometry.geometrical_parameters import Geometrical_Parameters

class Restraint:
    def __init__(self,pair1_index,pair2_index,force_constant,target_value,param_constant,en_restr_func='linear'):
        '''en_restr_func - 'elastic','linear' or callable'''
        self.pair1_index = pair1_index
        self.pair2_index = pair2_index
        self.force_constant = force_constant
        self.target_value = target_value
        self.param_constant = param_constant
        if en_restr_func == 'linear':
            en_restr_func = self._get_linear_restraint_energy
        elif en_restr_func == 'elastic':
            en_restr_func = self._get_elastic_restraint_energy
        self.get_restraint_energy = en_restr_func
    
    def _get_linear_restraint_energy(self,CG_structure):
        o1 = CG_structure.origins[self.pair1_index]
        o2 = CG_structure.origins[self.pair2_index]
        dist = (o1-o2).norm()
        return self.force_constant*dist/self.param_constant
        
    def _get_elastic_restraint_energy(self,all_coords):

        R1,o1 = all_coords.ref_frames[self.pair1_index],all_coords.origins[self.pair1_index]
        R2,o2 = all_coords.ref_frames[self.pair2_index],all_coords.origins[self.pair2_index]
        ref_frames = torch.stack([R2,R1])
        origins = torch.vstack([o2,o1]).reshape(-1,1,3)
        params = Geometrical_Parameters(origins=origins,ref_frames=ref_frames).local_params[1].to(self.target_value.device)
        params_dif = params - self.target_value
        dif_matrix = torch.matmul(params_dif.reshape(6,1), params_dif.reshape(1,6))
        return self.force_constant*torch.einsum('ij,ij',dif_matrix, self.param_constant)/2.0
    
    def to(self,device):
        self.target_value = self.target_value.to(device)
        self.param_constant = self.param_constant.to(device)