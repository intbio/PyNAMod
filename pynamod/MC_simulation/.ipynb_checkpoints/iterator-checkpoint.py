import torch
import time
import numpy as np
from tqdm.auto import tqdm

from pynamod.geometry.trajectories import Integrator_Trajectory

from pynamod.geometry.bp_step_geometry import Geometry_Functions

class Iterator:
    def __init__(self,dna_structure,energy,trajectory,sigma_transl=1,sigma_rot=1):
        self.dna_structure = dna_structure
        self.energy = energy
        self.sigma_transl = sigma_transl
        self.sigma_rot = sigma_rot
        self.h5_trajectory = trajectory
        self.h5_trajectory.attrs_names.append('energies')
    
    
    
    def run(self,movable_steps,target_accepted_steps=int(1e5),max_steps=int(1e6),transfer_to_memory_every=None,save_every=1,
            HDf5_file=None,mute=False,start_from_traj=False,KT_factor=1,integration_mod='minimize',device='cpu'):
        
        self._prepare_system(target_accepted_steps,transfer_to_memory_every,device,movable_steps,integration_mod,start_from_traj)
        
        total_step_bar = tqdm(total=max_steps,desc='Steps',disable=mute)
        info_pbar = tqdm(total=100,bar_format='{l_bar}{bar}{postfix}',desc='Acceptance rate',disable=mute)
        KT = KT_factor*self.energy.KT
        print("Start time:",time.strftime('%D %H:%M:%S',time.localtime()))
        try:
            while self.total_step < max_steps and self.accepted_steps < target_accepted_steps:
                
                change_index = self._get_cur_change_index(integration_mod)
                self._integration_step(KT,save_every,change_index)
                
                total_step_bar.update(1)
                info_pbar.set_postfix(E=f'{self.prev_e.sum().item():.2f}, Acc.st {self.accepted_steps}')
                info_pbar.set_description(f'Acceptance rate %')
                acceptance_rate=100*self.accepted_steps/self.total_step
                info_pbar.update(acceptance_rate-info_pbar.n)
                    
                    
        
        except KeyboardInterrupt:
            print('interrupted!')
        self._transfer_to_memory(steps=self.trajectory.cur_step)
        if self.accepted_steps >= target_accepted_steps:
            print('target accepted steps reached')
        print("Finish time:",time.strftime('%D %H:%M:%S',time.localtime()))
        print('it/s:', '%.2f'%total_step_bar.format_dict["rate"])
        print('accepted steps:',self.accepted_steps)
        print('total steps:',self.total_step)
    
    def to(self,device):
        self.trajectory.to(device)
        self.energy.to(device)
        
            
            
    def _prepare_system(self,target_accepted_steps,transfer_to_memory_every,device,movable_steps,integration_mod,start_from_traj):
        self.normal_scale = torch.tensor([*[self.sigma_transl]*3,*[self.sigma_rot]*3],device=device)
        
        if not transfer_to_memory_every:
            transfer_to_memory_every = target_accepted_steps
        self.transfer_to_memory_every = transfer_to_memory_every    
        self.total_step=0
        self.accepted_steps=0
        self.last_accepted=0
        
        self._create_tens_trajectory(start_from_traj)
        self.to(device)
        self.energy.mod_real_space_mat()
        self.prev_e = torch.hstack(self.energy.get_energy_components(self.trajectory))
        try:
            self.h5_trajectory.file[str(self.h5_trajectory.cur_step).zfill(self.h5_trajectory.string_format_val
                                                                          )].create_dataset('energies',data=self.prev_e.numpy(force=True))
        except ValueError:
            pass
        self.energy_comp_traj = torch.zeros(self.transfer_to_memory_every,4,device=device)
        
        self._set_change_indices(movable_steps,integration_mod)
        
        self.geom_func = Geometry_Functions()
        self.change = torch.zeros(6,dtype=self.trajectory.origins.dtype,device=device)
        self.normal_mean = torch.zeros(6,device=device)
    
    
    def _set_change_indices(self,movable_steps,integration_mod):
        self.movable_ind = torch.arange(self.trajectory.origins.shape[0],dtype=int)[movable_steps]
        if integration_mod == 'minimize':
            self.cur_movable_ind = 0
        elif integration_mod == 'random_step':
            self.movable_ind_len = self.movable_ind.shape[0]
    
    
    def _create_tens_trajectory(self,start_from_traj):
        if start_from_traj:
            init_local_params = torch.from_numpy(self.h5_trajectory._get_frame_attr('local_params'))
            init_ref_frames = torch.from_numpy(self.h5_trajectory._get_frame_attr('ref_frames'))
            init_ori = torch.from_numpy(self.h5_trajectory._get_frame_attr('origins'))
            init_prot_ori = torch.from_numpy(self.h5_trajectory._get_frame_attr('prot_origins'))
        else:
            init_local_params = self.dna_structure.dna.geom_params.local_params
            init_ref_frames = self.dna_structure.dna.geom_params.ref_frames
            init_ori = self.dna_structure.dna.geom_params.origins
            if self.dna_structure.proteins:
                init_prot_ori = torch.vstack([protein.origins for protein in self.dna_structure.proteins[::-1]])
            else:
                init_prot_ori = torch.empty((0,3))
        ln = init_ref_frames.shape[0]
        traj_len = self.transfer_to_memory_every+1
        dtype = init_ref_frames.dtype
        prot_origins_ln = init_prot_ori.shape[0]

        self.trajectory = Integrator_Trajectory(self.dna_structure.proteins,dtype,traj_len,ln)
        
        self.trajectory.origins,self.trajectory.ref_frames, = init_ori,init_ref_frames
        self.trajectory.local_params,self.trajectory.prot_origins = init_local_params,init_prot_ori
        if hasattr(self,'h5_trajectory'):
            self.h5_trajectory._set_frame_attr('origins',init_ori)
            self.h5_trajectory._set_frame_attr('prot_origins',init_prot_ori)
            self.h5_trajectory._set_frame_attr('ref_frames',init_ref_frames)
            self.h5_trajectory._set_frame_attr('local_params',init_local_params)
        
    def _integration_step(self,KT,save_every,linker_bp_index):
        prot_sl_index = self._apply_rotation(linker_bp_index)
        
#       static_origins = torch.vstack([self.trajectory.origins[:linker_bp_index],self.trajectory.prot_origins[prot_sl_index:]])
#         e_dif_components,e_mat,s_mat = self.energy.get_energy_dif(self.prev_e,static_origins,self.geom_func.origins[linker_bp_index:],
#                                                                   self.geom_func,linker_bp_index,prot_sl_index+self.trajectory.origins_traj.shape[1])
        
#         e_dif_components = torch.hstack(e_dif_components)
    
#        Del_E=e_dif_components.sum()
        e_cur = torch.hstack((self.energy.get_energy_components(self.geom_func,save_matr=False)))
        Del_E = (e_cur - self.prev_e).sum()
        r = torch.rand(1).item()
        self.total_step += 1

        if not Del_E.isnan() and Del_E < 0 or (not(torch.isinf(torch.exp(Del_E))) and (r  <= torch.exp(-Del_E/KT))):
            # self.energy.update_matrices(e_mat,s_mat,linker_bp_index,prot_sl_index+self.trajectory.origins_traj.shape[1])
            # self.prev_e += e_dif_components
            self.prev_e = e_cur
            self.accepted_steps += 1
            
            
            # if prot_sl_index == 0:
            #     self.trajectory.origins = self.geom_func.origins
            # else:
            #     static_prot_origins = self.trajectory.prot_origins[:prot_sl_index]
            #     self.trajectory.origins = self.geom_func.origins[:-prot_sl_index]
            #     self.trajectory.prot_origins = torch.vstack([static_prot_origins,self.geom_func.origins[-prot_sl_index:]]) 
            self.trajectory.origins = self.geom_func.origins[:self.trajectory.origins.shape[0]]
            self.trajectory.prot_origins = self.geom_func.origins[self.trajectory.origins.shape[0]:]
            self.trajectory.ref_frames = self.geom_func.ref_frames
            self.trajectory.local_params = self.geom_func.local_params
            
            if (self.accepted_steps% save_every) == 0:
                self.energy_comp_traj[self.trajectory.cur_step] = self.prev_e
                self.trajectory.cur_step += 1
                if self.trajectory.cur_step == self.transfer_to_memory_every:

                    self._transfer_to_memory()
                    self.trajectory.cur_step = 0
       
                # if prot_sl_index == 0:
                #     self.trajectory.origins = self.geom_func.origins
                # else:
                #     self.trajectory.origins = self.geom_func.origins[:-prot_sl_index]
                #     self.trajectory.prot_origins = torch.vstack([static_prot_origins,self.geom_func.origins[-prot_sl_index:]])
                self.trajectory.origins = self.geom_func.origins[:self.trajectory.origins.shape[0]]
                self.trajectory.prot_origins = self.geom_func.origins[self.trajectory.origins.shape[0]:]
                self.trajectory.ref_frames = self.geom_func.ref_frames
                self.trajectory.local_params = self.geom_func.local_params


            
    def _apply_rotation(self,linker_bp_index):   
        self.geom_func.ref_frames = self.trajectory.ref_frames.clone()
        self.trajectory.local_params[linker_bp_index] += torch.normal(mean=self.normal_mean,std=self.normal_scale,out=self.change)
        self.geom_func.local_params = self.trajectory.local_params.clone()
        prot_sl_index = self.trajectory.get_proteins_slice_ind(linker_bp_index)
        self.geom_func.origins = torch.vstack([self.trajectory.origins,
                                               self.trajectory.prot_origins[:prot_sl_index]
                                              ])
        self.geom_func.rotate_ref_frames_and_ori(linker_bp_index)
        self.geom_func.origins = torch.vstack([self.geom_func.origins,
                                               self.trajectory.prot_origins[prot_sl_index:]
                                              ])
        self.trajectory.local_params[linker_bp_index] -= self.change
        
        return prot_sl_index
        
        
    def _get_cur_change_index(self,integration_mod):
        if integration_mod == 'minimize':
            cur_index = self.cur_movable_ind
            self.cur_movable_ind += 1
            if self.cur_movable_ind == self.movable_ind.shape[0]:
                self.cur_movable_ind = 0
        elif integration_mod == 'random_step':
            cur_index = torch.randint(self.movable_ind_len,(1,))
            
        return self.movable_ind[cur_index]
    
    def _transfer_to_memory(self,steps=None):
        if not steps:
            steps = self.transfer_to_memory_every
        
        origins_traj = self.trajectory.origins_traj[:steps].numpy(force=True)
        ref_frames_traj = self.trajectory.ref_frames_traj[:steps].numpy(force=True)
        local_params_traj = self.trajectory.local_params_traj[:steps].numpy(force=True)
        prot_origins_traj = self.trajectory.prot_origins_traj[:steps].numpy(force=True)
        energy_comp_traj = self.energy_comp_traj[:steps].numpy(force=True)

        
        for i in range(steps):
            self.h5_trajectory.cur_step += 1
            self.h5_trajectory.add_frame(self.h5_trajectory.cur_step,origins=origins_traj[i],ref_frames = ref_frames_traj[i],
                                         local_params=local_params_traj[i],prot_origins=prot_origins_traj[i])
            try:
                self.h5_trajectory.file[str(self.h5_trajectory.cur_step).zfill(self.h5_trajectory.string_format_val)].create_dataset('energies',data=energy_comp_traj[i])
            except ValueError:
                self.h5_trajectory.file[str(self.h5_trajectory.cur_step).zfill(self.h5_trajectory.string_format_val)]['energies'][:] = energy_comp_traj[i]
    