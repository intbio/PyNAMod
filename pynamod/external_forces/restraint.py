import torch
from pynamod.geometry.geometrical_parameters import Geometrical_Parameters

class Restraint:
    def __init__(self,pair1_index,pair2_index,force_constant,target_step_parameters,max_deviation):
        self.pair1_index = pair1_index
        self.pair2_index = pair2_index
        self.force_constant = force_constant
        self.target_step_parameters = target_step_parameters
        self.max_deviation = max_deviation
        
    def get_restraint_energy(self,CG_structure):

        R1,o1 = CG_structure.dna.ref_frames[self.pair1_index],CG_structure.dna.origins[self.pair1_index]
        R2,o2 = CG_structure.dna.ref_frames[self.pair2_index],CG_structure.dna.origins[self.pair2_index]
        ref_frames = torch.stack([R1,R2])
        origins = torch.vstack([o1,o2]).reshape(-1,1,3)
        params = Geometrical_Parameters(origins=origins,ref_frames=ref_frames).local_params.to(self.target_step_parameters.device)
        energy = self.force_constant * torch.sum(((params - self.target_step_parameters)/self.max_deviation)**2)
            
        return energy
    
    def to(self,device):
        self.target_step_parameters = self.target_step_parameters.to(device)
        self.max_deviation = self.max_deviation.to(device)