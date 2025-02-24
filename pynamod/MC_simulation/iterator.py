import torch
import numpy as np
from tqdm.auto import tqdm

from pynamod.geometry.all_coords import All_Coords

class Iterator:
    def __init__(self,dna_structure,energy,sigma_transl=1,sigma_rot=1):
        self.dna_structure = dna_structure
        self.energy = energy
        self.energy.set_energy_matrices(self.dna_structure)
        self.sigma_transl = sigma_transl
        self.sigma_rot = sigma_rot
        
    
    def to(self,device):
        self.energy.to(device)
        self.dna_structure.to(device)
    
    
    def run(self,movable_steps,target_accepted_steps=1e5,max_steps=1e6,save_every=1,HDf5_file=None,mute=False,KT=1,integration_mod='minimize',device='cpu'):
        scale = torch.Tensor([*[self.sigma_transl]*3,*[self.sigma_rot]*3])
                
        
        self.prev_e = torch.finfo(float).max
        total_step_bar = tqdm(total=max_steps,desc='Steps')
        info_pbar = tqdm(total=100,bar_format='{l_bar}{bar}{postfix}',desc='Acceptance rate',disable=mute)
        self.total_step=0
        self.accepted_steps=0
        self.last_accepted=0
        self.dna_structure.all_coords.extend_trajectories(['local_params','origins','ref_frames'],target_accepted_steps/save_every)
        init_local_params,init_ref_frames,init_ori = self.dna_structure.all_coords.local_params,self.dna_structure.all_coords.ref_frames,self.dna_structure.all_coords.origins
        self.dna_structure.all_coords._cur_tr_step += 1
        self.dna_structure.all_coords.get_new_params_set(init_local_params,init_ref_frames,init_ori)
        if device == 'cuda':
            self.to('cuda')
        self.movable_ind = torch.arange(self.dna_structure.all_coords.len,dtype=int)[movable_steps]
        if integration_mod == 'minimize':
            self.cur_movable_ind = 0
        elif integration_mod == 'random_step':
            self.movable_ind_len = movable_ind.shape[0]
        self.energies = []
        self.change = torch.normal(mean=torch.zeros(max_steps,6),std=scale).to(device)
        try:
            while self.total_step < max_steps and self.accepted_steps < target_accepted_steps:
                
                change_index = self._get_cur_change_index(integration_mod)
                self._integration_step(scale,KT,save_every,change_index)
                total_step_bar.update(1)
                info_pbar.set_postfix(E=f'{self.prev_e.item():.2f}, Acc.st {self.accepted_steps}')
                info_pbar.set_description(f'Acceptance rate %')
                acceptance_rate=100*self.accepted_steps/self.total_step
                info_pbar.update(acceptance_rate-info_pbar.n)
                    
                    
        
        except KeyboardInterrupt:
            print('interrupted!')
        if self.accepted_steps >= target_accepted_steps:
            print('target accepted steps reached')
        else:
            saved_steps = self.accepted_steps // save_every
            self.dna_structure.all_coords._origins_traj = self.dna_structure.all_coords._origins_traj[:saved_steps+1]
            self.dna_structure.all_coords._ref_frames_traj = self.dna_structure.all_coords._ref_frames_traj[:saved_steps+1]
            self.dna_structure.all_coords._local_params_traj = self.dna_structure.all_coords._local_params_traj[:saved_steps+1]
            if self.dna_structure.all_coords._cur_tr_step == self.dna_structure.all_coords._origins_traj.shape[0]:
                self.dna_structure.all_coords._cur_tr_step -= 1
        print('accepted_steps',self.accepted_steps)
        
    def _integration_step(self,scale,KT,save_every,linker_bp_index):
        params,ref_frames,origins = self.dna_structure.all_coords.local_params.clone(),self.dna_structure.all_coords.ref_frames.clone(),self.dna_structure.all_coords.origins.clone()
        self.dna_structure.all_coords.local_params[linker_bp_index] += self.change[self.total_step]
        total_e = self.energy.get_full_energy(self.dna_structure)
        

        Del_E=total_e-self.prev_e
        r = torch.rand(1).item()
        self.total_step += 1

        if Del_E < 0 or (not(torch.isinf(torch.exp(Del_E))) and (r  <= torch.exp(-Del_E/KT))):
            self.prev_e = total_e
            self.accepted_steps += 1
            if (self.accepted_steps% save_every) == 0:
                params,ref_frames,origins = self.dna_structure.all_coords.local_params.clone(),self.dna_structure.all_coords.ref_frames.clone(),self.dna_structure.all_coords.origins.clone()
                if self.accepted_steps < self.dna_structure.all_coords._origins_traj.shape[0]-1:
                    self.dna_structure.all_coords._cur_tr_step += 1
                    self.dna_structure.all_coords.get_new_params_set(params,ref_frames,origins)
                self.energies.append(total_e.item())
                

        else:        
            self.dna_structure.all_coords.get_new_params_set(params,ref_frames,origins)   
        torch.cuda.synchronize()
        
    def _get_cur_change_index(self,integration_mod):
        if integration_mod == 'minimize':
            cur_index = self.cur_movable_ind
            self.cur_movable_ind += 1
            if self.cur_movable_ind == self.movable_ind.shape[0]:
                self.cur_movable_ind = 0
        elif integration_mod == 'random_step':
            cur_index = torch.randint(self.movable_ind_len,1)
            
        return self.movable_ind[cur_index]