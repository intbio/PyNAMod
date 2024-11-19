import torch
import numpy as np
from tqdm.auto import tqdm

class Iterator:
    def __init__(self,dna_structure,energy,sigma_transl=1,sigma_rot=1):
        self.dna_structure = dna_structure
        self.dna_structure.move_to_torch()
        self.energy = energy
        self.energy.set_energy_matrices(self.dna_structure)
        self.sigma_transl = sigma_transl
        self.sigma_rot = sigma_rot
        
    def run(self,movable_steps,target_accepted_steps=1e5,max_steps=1e6,save_every=10,HDf5_file=None,mute=False,KT=1):
        scale = [*[self.sigma_transl]*3,*[self.sigma_rot]*3]
                
        
        prev_e = torch.finfo(float).max
        if HDf5_file:
            file = h5py.File(HDf5_file,'w')
            data_length = target_accepted_steps//save_every
            frames_data = file.create_dataset('frames',(data_length,*current_frame.shape),dtype=np.float16, compression="gzip")
            saved_steps = file.create_dataset('saved_steps',(data_length,),dtype=int, compression="gzip")
            acceptance_rate_data = file.create_dataset('acceptance_rate',(1,),dtype=np.float16, compression="gzip")
            
            energies_data = file.create_group('energies')
            bend = energies_data.create_dataset('bend',(data_length,),dtype=np.float64, compression="gzip")
            real = energies_data.create_dataset('real',(data_length,),dtype=np.float64, compression="gzip")
            
            beads = file.create_group('beads')
            dna = beads.create_dataset('dna',(data_length,self.dna_structure.steps_params.shape[0],3),dtype=np.float16, compression="gzip")
            cg_beads = beads.create_dataset('cg_protein_beads',(data_length,self.dna_structure.get_proteins_attr('cg_radii').shape[0]),dtype=np.float16, compression="gzip")
            
        else:
            frames = []
            energies = []
            real_space = []
        total_step_bar = tqdm(total=max_steps,desc='Steps')
        info_pbar = tqdm(total=100,bar_format='{l_bar}{bar}{postfix}',desc='Acceptance rate',disable=mute)
        total_step=0
        accepted_steps=0
        last_accepted=0
        linker_bp_index_list = torch.arange(self.dna_structure.steps_params.shape[0],dtype=int)[movable_steps]
        
        try:
            while total_step < max_steps and accepted_steps < target_accepted_steps:


                for linker_bp_index in linker_bp_index_list:
                   
                    change = torch.normal(mean=torch.zeros(6),std=torch.Tensor(scale))
                    self.dna_structure.steps_params[linker_bp_index] += np.array(change)
                    total_e = self.energy.get_full_energy(self.dna_structure)

                    Del_E=total_e-prev_e
 
                    r = torch.rand(1).item()
                    total_step+=1
                    total_step_bar.update(1)
            
                    if Del_E < 0 or (not(torch.isinf(torch.exp(Del_E))) and (r  <= np.exp(-Del_E/KT))):
                        prev_e=total_e
                        accepted_steps+=1
                        if (accepted_steps% save_every) == 0:
                            
                            if HDf5_file:
                                data_index = accepted_steps// save_every - 1
                                frames_data[data_index,:,:] = current_frame
                                saved_steps[data_index] = accepted_steps
                                bend[data_index] = bend_e
                                real[data_index] = real_e
                                dna[data_index,:,:] = cur_ref_frames[:,3,:3]
                                ncp[data_index,:,:] = ncp_beads
                                file.flush()
                               
                            else:
                                frames.append(self.dna_structure.steps_params)
                                energies.append(total_e)
                                    
                        if accepted_steps == target_accepted_steps:
                            break
                    else:
                        self.dna_structure.steps_params[linker_bp_index] -= np.array(change)
                            
                    info_pbar.set_postfix(E=f'{prev_e:.2f}, Acc.st {accepted_steps}')
                    info_pbar.set_description(f'Acceptance rate %')
                    acceptance_rate=100*accepted_steps/total_step
                    info_pbar.update(acceptance_rate-info_pbar.n)
                    
        
        except KeyboardInterrupt:
            print('interrupted!')
        if accepted_steps >= target_accepted_steps:
            print('target accepted steps reached')
        print('accepted_steps',accepted_steps)
        if not HDf5_file:
            return(frames,energies)
        else:
            acceptance_rate_data[0] = acceptance_rate
            file.close()        
            
            
    def _integration_step(self,):
        pass