import torch

class Restraint:
    def __init__(self,pair1_index,pair2_index,force_constant,target_step_parameters,max_deviation):
        self.pair1_index = pair1_index
        self.pair2_index = pair2_index
        self.force_constant = force_constant
        self.target_step_parameters = target_step_parameters
        self.max_deviation = max_deviation
        
    def get_restraint_energy(self,DNA_structure):

        pair1 = DNA_structure.pairs_list[self.pair1_index]
        pair2 = DNA_structure.pairs_list[self.pair2_index]
        params = torch.Tensor(get_params_for_single_step_stock(pair2.om,pair1.om,pair2.Rm,pair1.Rm)[0])
        energy = self.force_constant * torch.sum(((params - self.target_step_parameters)/self.max_deviation)**2)
            
        return energy