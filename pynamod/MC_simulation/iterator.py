import torch
import numpy as np
from tqdm.auto import tqdm

from pynamod.geometry.all_coords import All_Coords

class Iterator:
    def __init__(self,dna_structure,energy,sigma_transl=1,sigma_rot=1):
        self.dna_structure = dna_structure
        #self.dna_structure.move_to_torch()
        self.energy = energy
        self.energy.set_energy_matrices(self.dna_structure)
        self.sigma_transl = sigma_transl
        self.sigma_rot = sigma_rot
        
    def run(self,movable_steps,target_accepted_steps=1e5,max_steps=1e6,save_every=1,HDf5_file=None,mute=False,KT=1,integration_mod='minimize'):
        scale = torch.Tensor([*[self.sigma_transl]*3,*[self.sigma_rot]*3])
                
        
        self.prev_e = torch.finfo(float).max
        total_step_bar = tqdm(total=max_steps,desc='Steps')
        info_pbar = tqdm(total=100,bar_format='{l_bar}{bar}{postfix}',desc='Acceptance rate',disable=mute)
        self.total_step=0
        self.accepted_steps=0
        self.last_accepted=0
        self.all_coords = All_Coords(self.dna_structure,int(target_accepted_steps))
        movable_ind = torch.arange(self.all_coords.dna_len,dtype=int)[movable_steps]
        if integration_mod == 'minimize':
            linker_bp_index_list = movable_ind
        elif integration_mod == 'random_step':
            linker_bp_index_list = torch.randint(torch.max(movable_ind),1)
        self.energies = []
        try:
            while self.total_step < max_steps and self.accepted_steps < target_accepted_steps:

                for linker_bp_index in linker_bp_index_list:
                    self._integration_step(scale,KT,save_every,linker_bp_index)
                    total_step_bar.update(1)
                    info_pbar.set_postfix(E=f'{self.prev_e.item():.2f}, Acc.st {self.accepted_steps}')
                    info_pbar.set_description(f'Acceptance rate %')
                    acceptance_rate=100*self.accepted_steps/self.total_step
                    info_pbar.update(acceptance_rate-info_pbar.n)
                    if self.accepted_steps == target_accepted_steps:
                        break
                
                if integration_mod == 'random_step':
                    linker_bp_index_list = torch.randint(torch.max(movable_ind),1)
        
        except KeyboardInterrupt:
            print('interrupted!')
        if self.accepted_steps >= target_accepted_steps:
            print('target accepted steps reached')
        print('accepted_steps',self.accepted_steps)           

        
    def _integration_step(self,scale,KT,save_every,linker_bp_index):
        change = torch.normal(mean=torch.zeros(6),std=scale)
        params,ref_frames,origins = self.all_coords.local_params.clone(),self.all_coords.ref_frames.clone(),self.all_coords.origins.clone()
        self.all_coords.local_params[linker_bp_index] += change
        total_e = self.energy.get_full_energy(self.all_coords,self.dna_structure)

        Del_E=total_e-self.prev_e

        r = torch.rand(1).item()
        self.total_step += 1

        if Del_E < 0 or (not(torch.isinf(torch.exp(Del_E))) and (r  <= torch.exp(-Del_E/KT))):
            self.prev_e = total_e
            self.accepted_steps += 1
            if (self.accepted_steps% save_every) == 0:

                self.all_coords._cur_tr_step += 1
                self.energies.append(total_e.item())

        else:
            self.all_coords._auto_rebuild_sw = False
            self.all_coords.local_params,self.all_coords.ref_frames,self.all_coords.origins = params,ref_frames,origins
            self.all_coords._auto_rebuild_sw = True
