import torch
import numpy as np
from tqdm.auto import tqdm

from pynamod.geometry.all_coords import All_Coords
from pynamod.geometry.trajectories import Tensor_Trajectory

class Iterator:
    def __init__(self,dna_structure,energy,trajectory,sigma_transl=1,sigma_rot=1):
        self.dna_structure = dna_structure
        self.energy = energy
        self.sigma_transl = sigma_transl
        self.sigma_rot = sigma_rot
        self.trajectory = trajectory
    
    
    
    def run(self,movable_steps,target_accepted_steps=int(1e5),max_steps=int(1e6),transfer_to_memory_every=None,save_every=1,
            HDf5_file=None,mute=False,KT=1,integration_mod='minimize',device='cpu',trajectory = None):
        
        self._prepare_system(trajectory,transfer_to_memory_every,device,movable_steps,integration_mod)
        
        total_step_bar = tqdm(total=max_steps,desc='Steps')
        info_pbar = tqdm(total=100,bar_format='{l_bar}{bar}{postfix}',desc='Acceptance rate',disable=mute)
        
        try:
            while self.total_step < max_steps and self.accepted_steps < target_accepted_steps:
                
                change_index = self._get_cur_change_index(integration_mod)
                self._integration_step(KT,save_every,change_index)
                
                total_step_bar.update(1)
                info_pbar.set_postfix(E=f'{self.prev_e.item():.2f}, Acc.st {self.accepted_steps}')
                info_pbar.set_description(f'Acceptance rate %')
                acceptance_rate=100*self.accepted_steps/self.total_step
                info_pbar.update(acceptance_rate-info_pbar.n)
                    
                    
        
        except KeyboardInterrupt:
            print('interrupted!')
        if self.accepted_steps >= target_accepted_steps:
            print('target accepted steps reached')

        print('accepted_steps',self.accepted_steps)
    
    def to(self,device):
        self.intg_all_coords.to(device)
        self.energy.to(device)
        
    
    def _remove_frames(self):
        if self.dna_structure.all_coords._cur_tr_step == self.dna_structure.all_coords.origins_traj.shape[0]:
            self.dna_structure.all_coords._cur_tr_step -= 1
        self.dna_structure.all_coords._origins_traj = self.dna_structure.all_coords._origins_traj[:self.dna_structure.all_coords._cur_tr_step+1]
        self.dna_structure.all_coords._ref_frames_traj = self.dna_structure.all_coords._ref_frames_traj[:self.dna_structure.all_coords._cur_tr_step+1]
        self.dna_structure.all_coords._local_params_traj = self.dna_structure.all_coords._local_params_traj[:self.dna_structure.all_coords._cur_tr_step+1]
            
            
    def _prepare_system(self,trajectory,transfer_to_memory_every,device,movable_steps,integration_mod):
        self.normal_scale = torch.tensor([*[self.sigma_transl]*3,*[self.sigma_rot]*3],device=device)
        
        if not transfer_to_memory_every:
            transfer_to_memory_every = target_accepted_steps
        self.transfer_to_memory_every = transfer_to_memory_every        
        
        self.total_step=0
        self.accepted_steps=0
        self.last_accepted=0
        
        self._create_all_coords()
        self.to(device)
        
        self.prev_e = self.energy.get_full_energy(self.intg_all_coords)
        
        self._set_change_indices(movable_steps,integration_mod)
        
        self.energies = []
        self.change = torch.zeros(6,dtype=self.dna_structure.all_coords.dtype,device=device)
        self.normal_mean = torch.zeros(6,device=device)
    
    
    def _set_change_indices(self,movable_steps,integration_mod):
        self.movable_ind = torch.arange(self.dna_structure.all_coords.len,dtype=int)[movable_steps]
        if integration_mod == 'minimize':
            self.cur_movable_ind = 0
        elif integration_mod == 'random_step':
            self.movable_ind_len = self.movable_ind.shape[0]
    
    
    def _create_all_coords(self):
        init_local_params,init_ref_frames,init_ori = self.dna_structure.all_coords.local_params,self.dna_structure.all_coords.ref_frames,self.dna_structure.all_coords.origins
        self.intg_all_coords = All_Coords(empty=True)
        
        dtype = self.dna_structure.all_coords.dtype
        origins_traj = torch.zeros(self.transfer_to_memory_every,init_ori.shape[0],1,3,dtype=dtype)
        ln = self.dna_structure.all_coords.len
        ref_frames_traj = torch.zeros(self.transfer_to_memory_every,ln,3,3,dtype=dtype)
        local_params_traj = torch.zeros(self.transfer_to_memory_every,ln,6,dtype=dtype)
        origins_traj[0],ref_frames_traj[0],local_params_traj[0] = init_ori,init_ref_frames,init_local_params
        self.intg_all_coords.len = ln
        self.intg_all_coords.trajectory = Tensor_Trajectory(None,None,None,self.intg_all_coords,ref_frames_traj=ref_frames_traj,
                                                            origins_traj=origins_traj,local_params_traj=local_params_traj)

    def _integration_step(self,KT,save_every,linker_bp_index):
        params,ref_frames,origins = self.intg_all_coords.local_params.clone(),self.intg_all_coords.ref_frames.clone(),self.intg_all_coords.origins.clone()
        self.intg_all_coords.local_params[linker_bp_index] += torch.normal(mean=self.normal_mean,std=self.normal_scale,out=self.change)
        
        total_e = self.energy.get_full_energy(self.intg_all_coords)

        Del_E=total_e-self.prev_e
        r = torch.rand(1).item()
        self.total_step += 1

        if Del_E < 0 or (not(torch.isinf(torch.exp(Del_E))) and (r  <= torch.exp(-Del_E/KT))):
            self.prev_e = total_e
            self.accepted_steps += 1
            if (self.accepted_steps% save_every) == 0:
                params,ref_frames,origins = self.intg_all_coords.local_params.clone(),self.intg_all_coords.ref_frames.clone(),self.intg_all_coords.origins.clone()
                self.intg_all_coords.trajectory.cur_step += 1
                if self.intg_all_coords.trajectory.cur_step == self.transfer_to_memory_every:
                    #self._transfer_to_memory()
                    self.intg_all_coords.trajectory.cur_step = 0
                    
                    
                self.intg_all_coords.get_new_params_set(params,ref_frames,origins)
                self.energies.append(total_e.item())
                

        else:
            self.intg_all_coords.get_new_params_set(params,ref_frames,origins)   
        torch.cuda.synchronize()
        
    def _get_cur_change_index(self,integration_mod):
        if integration_mod == 'minimize':
            cur_index = self.cur_movable_ind
            self.cur_movable_ind += 1
            if self.cur_movable_ind == self.movable_ind.shape[0]:
                self.cur_movable_ind = 0
        elif integration_mod == 'random_step':
            cur_index = torch.randint(self.movable_ind_len,(1,))
            
        return self.movable_ind[cur_index]
    
    def _transfer_to_memory(self):
        origins_traj = self.intg_all_coords.trajectory.origins_traj.cpu().numpy()
        ref_frames_traj = self.intg_all_coords.trajectory.ref_frames_traj.cpu().numpy()
        local_params_traj = self.intg_all_coords.trajectory.local_params_traj.cpu().numpy()
        for i in range(self.transfer_to_memory_every):
            self.trajectory.cur_step += 1
            self.trajectory.add_frame(self.trajectory.cur_step,origins=origins_traj[i],ref_frames = ref_frames_traj[i],local_params=local_params_traj[i])
    