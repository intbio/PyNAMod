

class Iterator:
    def __init__(self,dna_structure,energy):
        self.dna_structure = dna_structure
        self.energy = energy
        
        
    def run(self,target_accepted_steps,max_steps): 
        try:
            while total_step < max_steps and accepted_steps < target_accepted_steps:


                for linker_bp_index in linker_bp_index_list:
                   
                    temp_frame = self.dna_structure.copy()
                    temp_frame[linker_bp_index,6:]+=np.random.normal(scale=scale,size=[6])
                    sub_frame=get_bpstep_frame(temp_frame,movable_steps)
                    bend_e=_calc_bend_energy(sub_frame,force_matrix,average_bpstep_frame)

                    if accepted_steps%rebuild_index == 0:
                        dna_beads,ncp_beads,misc_beads,cur_ref_frames = self.get_all_beads_on_fiber(temp_frame,
                                                                                 linker_bp_coarse_index_list)
                    else:
                        dna_beads,ncp_beads,misc_beads,cur_ref_frames = self.get_all_beads_on_fiber_no_rebuild(ncp_beads,misc_beads,prev_ref_frames,
                                                         temp_frame[linker_bp_index],linker_bp_index,linker_bp_coarse_index_list)
                    if misc_beads is None:
                        all_beads=np.vstack((dna_beads,ncp_beads))
                    else:
                        all_beads=np.vstack((dna_beads,ncp_beads,misc_beads))
                    real_e,components=calc_real_space_total_energy(all_beads,**force_params,**energy_dict)
                    total_e=real_e+K_bend*bend_e
                    Del_E=total_e-prev_e

                    r = np.random.uniform(0,1)
                    total_step+=1
                    total_step_bar.update(1)
                    if Del_E < 0 or (not(np.isinf(np.exp(Del_E))) and (r  <= np.exp(-Del_E/KT))):
                        prev_e=total_e
                        accepted_steps+=1
                        prev_ref_frames = cur_ref_frames                     

                        current_frame=temp_frame.copy()
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
                                frames.append(current_frame)
                                components['bend']=bend_e
                                components['entities']=copy.deepcopy(self.entities)

                                components['linker_mask']=copy.deepcopy(np.sort(linker_bp_index_list))
                                energies.append(components)
                            if save_beads:
                                real_space.append({'dna':dna_beads,'ncp':ncp_beads,'misc':misc_beads})
                                    
                        if accepted_steps == target_accepted_steps:
                            break
                            
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