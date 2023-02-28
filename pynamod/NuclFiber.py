import requests 
import io
import pynucl
from pynucl.seq_utils import residues_3to1
from pynamod.energy_constants import * 
from pynamod.visual_ngl import show_ref_frames
from pynamod.non_DNA_geometry import get_obj_orientation_and_location,get_rotation_and_offset_ref_frame_to_obj
from pynamod.bp_step_geometry import rebuild_by_full_par_frame,rotate_bp_frames,get_ori_and_mat_from_step,rotate_origins
from pynamod.energy_funcs import *
from pynamod.energy_funcs import _calc_bend_energy
from pynamod.utils import get_movable_steps
from tqdm.auto import tqdm,trange
import threading
import copy

from six import StringIO
import MDAnalysis as mda
from MDAnalysis.analysis import align
from multiprocessing import Pool
import itertools
from MDAnalysis.coordinates.memory import MemoryReader

import logging

from pynamod.utils import W_C_pair
# надо добавить возможность "регуляризации фибрилл" то есть, чтобы можно было задавать "примерные" позиции диад, а они были уточнены, чтобы формировать конформационно допустимые модели
class Fiber_model(object):
    def __init__(self, sequence=None, ncp_pdb_ids=['3LZ0'],
                 ncp_dyad_locations=[],linker_lengths=None,linker_sequence=None,
                 entry_unwraps=None,exit_unwraps=None,linker_initial_model='olson_98',
                 loggingLevel=logging.INFO):
        '''
        sequence dominates over N+linker sequence
        linker_initial_model = 'olson_98' or 'bdna'
        '''
        logging.getLogger().setLevel(loggingLevel)
        self.entities={'NCP':{},'misc':{}}
        self.ncp_pdb_ids=ncp_pdb_ids=[ncp_pdb_id.lower().strip() for ncp_pdb_id in ncp_pdb_ids]
        self.ncp_bp_step_frames={}
        self.ncp_sequences={}
        self.ncp_offsets={}
        self.restrains=[]
        logging.info('Started loading structures from RCSB')
        for ncp_pdb_id in set(self.ncp_pdb_ids):            
            self.ncp_offsets[ncp_pdb_id], self.ncp_sequences[ncp_pdb_id], self.ncp_bp_step_frames[ncp_pdb_id] = load_ncp(ncp_pdb_id)

        if isinstance(sequence,str) and set(sequence.upper()).issubset(set('ATGC')):
             
            self.sequence=sequence.upper()
            if not (linker_sequence is None):
                logging.warning('linker_sequence ignored as sequence provided')
            
        elif not (linker_sequence is None):
            if not linker_lengths is None:
                logging.warning('linker_lengths ignored as linker_sequence provided')
            linker_lengths=len(linker_sequence)
            if (len(self.ncp_pdb_ids)>0):
                self.sequence=linker_sequence.upper()
                for i,ncp_pdb_id in enumerate(self.ncp_pdb_ids):
                    self.sequence+=self.ncp_sequences[ncp_pdb_id]
                    self.sequence+=linker_sequence.upper()                    
                
        else:
            logging.warning('No DNA sequence provided, breaking')
            return            
        
        self.initial_bp_step_frame=np.tile(BDNA_step,(len(self.sequence),1))
        self.pairs=[f'{n}-{W_C_pair[n]}' for n in self.sequence]
        if linker_initial_model=='olson_98':
            averages=get_consts_olson_98()[0]
            for i,(first,second) in enumerate(zip(self.sequence[:-1],self.sequence[1:]),1):
                self.initial_bp_step_frame[i,6:]=averages[f'{first}{second}']
        self.initial_bp_step_frame[0,6:] = 0
        
        if (linker_lengths==None ) and (len(self.ncp_pdb_ids)==len(ncp_dyad_locations)):
            self.place_ncps_on_fiber_by_dyad_pos(self.ncp_pdb_ids,ncp_dyad_locations,
                                                entry_unwraps=entry_unwraps,exit_unwraps=exit_unwraps)
        elif (linker_lengths==None ) and (len(self.ncp_pdb_ids)!=len(ncp_dyad_locations)):
            self.place_ncps_on_fiber_by_dyad_pos(self.ncp_pdb_ids,ncp_dyad_locations,
                                                entry_unwraps=entry_unwraps,exit_unwraps=exit_unwraps)
            logging.warning('ncp_pdb_ids and ncp_dyad_locations does not match, or no linker lengths provided')
        elif isinstance(linker_lengths,int) or (len(linker_lengths) == (len(self.ncp_pdb_ids)+1)):
            logging.warning('ncp_dyad_locations ignored as linker_sequence or linker_lengths provided')
            self.place_ncps_on_fiber_by_linker_lengths(self.ncp_pdb_ids,linker_lengths,
                                                      entry_unwraps=entry_unwraps,exit_unwraps=exit_unwraps)
            
        else:
            logging.warning('ncp_pdb_ids and ncp_dyad_locations does not match, or no linker lengths provided')
        
    def place_ncps_on_fiber_by_linker_lengths(self,ncp_pdb_ids,linker_lengths,
                                              entry_unwraps=None,exit_unwraps=None):
        self.ncp_pdb_ids=ncp_pdb_ids=[ncp_pdb_id.lower().strip() for ncp_pdb_id in ncp_pdb_ids]
        
        ncp_dyad_locations=[]
        i=0
        for j,ncp_id in enumerate(ncp_pdb_ids):
            if not ncp_id in self.ncp_bp_step_frames.keys():
                self.ncp_offsets[ncp_id], self.ncp_sequences[ncp_id],self.ncp_bp_step_frames[ncp_id]=load_ncp(ncp_id)
            if isinstance(linker_lengths,int):                
                offset0=self.ncp_bp_step_frames[ncp_id].index[0]
                ncp_dyad_locations.append(i+linker_lengths-offset0)
                i+=linker_lengths+self.ncp_bp_step_frames[ncp_id].shape[0]
            else:
                offset0=self.ncp_bp_step_frames[ncp_id].index[0]
                ncp_dyad_locations.append(i+linker_lengths[j]-offset0)
                i+=linker_lengths[j]+self.ncp_bp_step_frames[ncp_id].shape[0]
        self.place_ncps_on_fiber_by_dyad_pos(self.ncp_pdb_ids,ncp_dyad_locations,
                                             entry_unwraps=entry_unwraps,exit_unwraps=exit_unwraps)
        
    def place_ncps_on_fiber_by_dyad_pos(self,ncp_pdb_ids,ncp_dyad_locations,entry_unwraps=None,exit_unwraps=None):
        #TODO offsets must be rethinked
        self.entities['NCP']={}
        self.ncp_dyad_locations=ncp_dyad_locations
        
        if (entry_unwraps is None): entry_unwraps = [0]*len(ncp_dyad_locations)
        assert len(entry_unwraps) == len(ncp_dyad_locations)
        if (exit_unwraps is None): exit_unwraps = [0]*len(ncp_dyad_locations)
        assert len(exit_unwraps) == len(ncp_dyad_locations)
        
        self.ncp_pdb_ids = ncp_pdb_ids = [ncp_pdb_id.lower().strip() for ncp_pdb_id in ncp_pdb_ids]
        self.bp_step_frame = self.initial_bp_step_frame.copy()
        self.linker_mask = np.ones(self.bp_step_frame.shape[0]).astype(bool)
        
        for i,(ncp_id,dyad_loc,entry_unwrap,exit_unwrap) in enumerate(zip(ncp_pdb_ids,ncp_dyad_locations,
                                                 entry_unwraps,exit_unwraps)):
            if not ncp_id in self.ncp_bp_step_frames.keys():
                self.ncp_offsets[ncp_id],
                self.ncp_sequences[ncp_id],
                self.ncp_bp_step_frames[ncp_id]=load_ncp(ncp_id)
            offset0=self.ncp_bp_step_frames[ncp_id].index[0]+entry_unwrap
            offset1=self.ncp_bp_step_frames[ncp_id].index[-1]-exit_unwrap

            ncp_frame_np=self.ncp_bp_step_frames[ncp_id].to_numpy()
            if ((dyad_loc+offset0) >= 0) and ((dyad_loc+offset1) < len(self.sequence)):

                self.bp_step_frame[dyad_loc+offset0+1:dyad_loc+offset1+1,6:]=ncp_frame_np[entry_unwrap+1:(-exit_unwrap if exit_unwrap!= 0 else None), 6:]

                self.linker_mask[dyad_loc+offset0:dyad_loc+offset1+1]=False
                self.entities['NCP'][i]={'name':f'ncp_{i}',
                                         'loc' :dyad_loc,
                                         'entry_unwrap' :entry_unwrap,
                                         'left_dyad_offset':self.ncp_bp_step_frames[ncp_id].index[0],
                                         'right_dyad_offset':self.ncp_bp_step_frames[ncp_id].index[-1],
                                         'exit_unwrap' :exit_unwrap,
                                         'kind':'NCP',
                                         'type':ncp_id,
                                         'v_offset':self.ncp_offsets[ncp_id][1],
                                         'r_mat'   :self.ncp_offsets[ncp_id][0],
                                         'R':45}
                
                
            else:
                logging.warning(f' NCP #{i} {ncp_id} located on {dyad_loc} is out of sequence bounds, skipped')
                
    def move_ncp(self,ncp_entity_index,step,straight_dna=False,compensate_twist_steps=None):
        dyad_loc = self.entities['NCP'][ncp_entity_index]['loc']
        ncp_id = self.entities['NCP'][ncp_entity_index]['type']
        entry_unwrap = self.entities['NCP'][ncp_entity_index]['entry_unwrap'] 
        exit_unwrap = self.entities['NCP'][ncp_entity_index]['exit_unwrap']
        offset0=self.ncp_bp_step_frames[ncp_id].index[0]+entry_unwrap
        offset1=self.ncp_bp_step_frames[ncp_id].index[-1]-exit_unwrap
        ncp_frame_np = self.ncp_bp_step_frames[ncp_id].to_numpy()
        old_mask = self.linker_mask[dyad_loc+offset0:dyad_loc+offset1+1]
        if straight_dna:
            self.bp_step_frame[dyad_loc+offset0+1:dyad_loc+offset1+1,6:] = BDNA_step[6:]
        dyad_loc+=step
        self.bp_step_frame[dyad_loc+offset0+1:dyad_loc+offset1+1,6:]=ncp_frame_np[entry_unwrap+1:(-exit_unwrap if exit_unwrap!= 0 else None), 6:]
        if not compensate_twist_steps is None:
            self.bp_step_frame[dyad_loc+offset0+1-compensate_twist_steps:dyad_loc+offset0+1, 11]-= (step*35.666)/compensate_twist_steps
            self.bp_step_frame[dyad_loc+offset1+1:+dyad_loc+offset1+1+compensate_twist_steps,11]+= (step*35.666)/compensate_twist_steps
        self.linker_mask[dyad_loc+offset0:dyad_loc+offset1+1]=old_mask
        self.entities['NCP'][ncp_entity_index]['loc']=dyad_loc
                
    def add_restraint(self,a_num,b_num,dist,a_kind='ncp',b_kind='ncp'):
        restraint={'a':[a_kind,a_num],'b':[b_kind,b_num],'dist':dist}
        if not (restraint in self.restrains):
            self.restrains.append({'a':[a_kind,a_num],'b':[b_kind,b_num],'dist':dist})
    
    def view_fiber(self,bp_step_frame=None,spheres=True,arrows=False,diamonds=False,boxes=False,
                   show_model=False,show_misc=True):
        '''
        VIEW restraints
        ребилд нужен бы еще атомарный
        '''
        bp_step_frame = self.bp_step_frame if bp_step_frame is None else bp_step_frame
        self.cur_ref_frames=rebuild_by_full_par_frame_numba(bp_step_frame)
        view=show_ref_frames(self.cur_ref_frames,spheres=spheres,arrows=arrows,diamonds=diamonds,boxes=boxes,
                            bp_colors_mask=self.linker_mask)
        self.ncp_beads=np.zeros((len(self.entities['NCP']),3))
        cylynders=np.zeros((len(self.entities['NCP']),2,3))
        
        linker_bp_index_list,linker_bp_coarse_index_list=get_linkers_bp_indexes(self.linker_mask)
        dna_beads,self.ncp_beads,misc_beads,cur_ref_frames=self.get_all_beads_on_fiber(bp_step_frame,linker_bp_coarse_index_list)
        print(dna_beads.shape)   
    
        view.shape.add_buffer('sphere',position=self.ncp_beads.flatten().tolist(),
                                  color=[1,0,0]*self.ncp_beads.shape[0],
                                  radius=  [40]*self.ncp_beads.shape[0])
        if show_model:
            view.shape.add_buffer('sphere',position=dna_beads.flatten().tolist(),
                                  color=[0,1,0]*dna_beads.shape[0],
                                  radius=  [10]*dna_beads.shape[0])
        if (not (misc_beads is None)) and show_misc:
            view.shape.add_buffer('sphere',position=misc_beads.flatten().tolist(),
                              color=[0,1,1]*misc_beads.shape[0],
                              radius=  [15]*misc_beads.shape[0])
        
#         elif ncp_repr=='cylinder':
#                     view.shape.add_buffer('cylinder',
#                               position1=cylynders[:,0].flatten().tolist(),
#                               position2=cylynders[:,1].flatten().tolist(),
#                               color=[1,0,0]*(self.ncp_beads.shape[0]),radius=[30]*self.ncp_beads.shape[0])
        return(view)
    
    
    
    def get_all_beads_on_fiber(self,bp_step_frame, linker_bp_coarse_index_list=None,
                               linker_mask=None, entities=None):
        cur_ref_frames=rebuild_by_full_par_frame(bp_step_frame)
        if linker_mask is None:
            linker_mask = self.linker_mask
        if entities is None:
            entities = self.entities
        ### DNA BEADS ###
        if linker_bp_coarse_index_list is None:
            linker_bp_coarse_index_list=linker_mask
        dna_beads=cur_ref_frames[linker_bp_coarse_index_list,3,:3]
        
        ### NCP BEADS ###
        ncp_beads=np.zeros((len(entities['NCP']),3))           
        for num,NCP in entities['NCP'].items():
            ref_mat=cur_ref_frames[NCP['loc'],:3,:3]
            o1=cur_ref_frames[NCP['loc'],3,:3]
            R2,of_vec=NCP['r_mat'],NCP['v_offset']
            calcR2,calcO2=get_obj_orientation_and_location(ref_mat,o1,R2,of_vec)
            ncp_beads[num]=calcO2
         ### MISC BEADS ###
        if len(entities['misc'])!=0:
            misc_beads=np.zeros((len(entities['misc']),3))           
            for num,NCP in entities['misc'].items():
                ref_mat=cur_ref_frames[NCP['loc'],:3,:3]
                o1=cur_ref_frames[NCP['loc'],3,:3]
                R2,of_vec=NCP['r_mat'],NCP['v_offset']
                calcR2,calcO2=get_obj_orientation_and_location(ref_mat,o1,R2,of_vec)
                misc_beads[num]=calcO2
        else:
            misc_beads=None
        return(dna_beads,ncp_beads,misc_beads,cur_ref_frames)
    
    
    def get_all_beads_on_fiber_no_rebuild(self,prev_ncp_beads,prev_misc_beads,prev_ref_frames,
                                          changed_step_frame,change_index, linker_bp_coarse_index_list=None,
                               linker_mask=None, entities=None):
        cur_ref_frames,rot_mat,ref_ori0,ref_ori1=rotate_bp_frames(copy.deepcopy(prev_ref_frames),changed_step_frame,change_index)
        if linker_mask is None:
            linker_mask = self.linker_mask
        if entities is None:
            entities = self.entities
        ### DNA BEADS ###
        if linker_bp_coarse_index_list is None:
            linker_bp_coarse_index_list=linker_mask
        dna_beads=cur_ref_frames[linker_bp_coarse_index_list,3,:3]
        
        ### NCP BEADS ###
        ncp_beads= rotate_origins(prev_ncp_beads.copy(),rot_mat,ref_ori0,ref_ori1)
         ### MISC BEADS ###
        if len(entities['misc'])!=0:
            misc_beads=rotate_origins(prev_misc_beads.copy(),rot_mat,ref_ori0,ref_ori1)
        else:
            misc_beads=None
        return(dna_beads,ncp_beads,misc_beads,cur_ref_frames)
        
        
    def get_energy(self,bending=True,excluded=True,
                   electrostatic=False,dist_pairs=True,energy_dict={}):
        '''
        energy_dict={
        'dna_r':5,'ncp_r':60,'misc_r':5,
        'dna_eps':0.5,'ncp_eps':1,'misc_eps':0,
        'dna_q':0,'ncp_q':0,'misc_q':0,
        'K_excl':1,'K_elec':1,'K_dist':1}
        '''
        #bending
        if not 'K_bend' in energy_dict:
            energy_dict['K_bend']=1
        AVERAGE,FORCE_CONST,DISP=get_consts_olson_98()
        movable_steps=np.argwhere(self.linker_mask).flatten()[1:]-1
        #movable_steps=get_movable_steps(movable_steps)
        
        bend_e = energy_dict['K_bend'] * get_bend_energy(self.bp_step_frame,self.pairs,
                                                         movable_steps,FORCE_CONST,AVERAGE)
        # real space
        
        linker_bp_index_list,linker_bp_coarse_index_list=get_linkers_bp_indexes(self.linker_mask)
        dna_beads,ncp_beads,misc_beads=self.get_all_beads_on_fiber(self.bp_step_frame,linker_bp_coarse_index_list)

        if misc_beads is None:
            all_beads=np.vstack((dna_beads,ncp_beads))
        else:
            all_beads=np.vstack((dna_beads,ncp_beads,misc_beads))
        if dist_pairs:
            force_params = get_real_space_force_mat(dna_beads,ncp_beads,misc_beads,
                                                    dist_pairs=self.restrains,**energy_dict)
        else:
            force_params=get_real_space_force_mat(dna_beads,ncp_beads,misc_beads,**energy_dict)
            
        real_space_e=calc_real_space_total_energy(all_beads,**force_params,**energy_dict)
        real_space_e[1]['bend']=bend_e
        return(real_space_e[0]+bend_e,real_space_e[1])
    
#     async def _start_mc_by_linker_async(self,**kwargs):
#         while break_cicle:

    
    def start_mc_by_linker_new(self,bending=True,excluded=True,
                   electrostatic=False,dist_pairs=True,energy_dict={},
                           KT=1,sigma_transl=1,sigma_rot=1,
                           target_accepted_steps=200,max_steps=10000,
                           mute=False,save_every=1,save_beads=False,ncp_remodel_dict=None,rebuild_index = 50):
        '''
        TODO 
        rebuild after init
        SAVE METAINFO OF START INCLUDING THE DATE
        show graphs with energy and stuff
        SAVE AS MDA TRAJ
        WRITE DIRECTLY INTO OBJECT TO USE LATTER
        ncp_remodel_dict : {macrocycle_step:50,target_dist:160,step_size:1,compensate_twist_steps:None}
        режимы:
        линкеры по очереди
        случайные линкеры
        нуклеотиды по очереди - тут возможен чит, когда не надо пересчитывать все, а поворачивать все биды (должно быть быстрее, чем пересчитывать все с нуля)
         - случайные 6 параметров
         - случайный 1 параметр
         случайные нуклеотиды
         - случайные 6 параметров
         - случайный 1 параметр
        energy_dict={
        'dna_r':5,'ncp_r':60,'misc_r':5,
        'dna_eps':0.5,'ncp_eps':1,'misc_eps':0,
        'dna_q':0,'ncp_q':0,'misc_q':0,
        'K_excl':1,'K_elec':1,'K_dist':1,'K_bend':1}
        https://en.wikibooks.org/wiki/Molecular_Simulation/Monte_Carlo_Methods
        '''
        scale = [*[sigma_transl]*3,*[sigma_rot]*3] # 6 width coefficients for random step sampling
        
        #setting up DNA bp space
        # TODO isolate all energy terms to separate structure (init function for all energy related terms)
        if not 'K_bend' in energy_dict:
            energy_dict['K_bend'] = 1
        K_bend=energy_dict['K_bend']
        AVERAGE,FORCE_CONST,DISP = get_consts_olson_98()
        
        # todo fixme 
        movable_steps = np.argwhere(self.linker_mask).flatten()[1:] - 1 

        force_matrix = get_force_matrix(self.pairs, movable_steps, FORCE_CONST)
        average_bpstep_frame = get_force_matrix(self.pairs, movable_steps, AVERAGE)        
        
        # setting up real space
        linker_bp_index_list,linker_bp_coarse_index_list = get_linkers_bp_indexes(self.linker_mask)
        dna_beads,ncp_beads,misc_beads,cur_ref_frames = self.get_all_beads_on_fiber(self.bp_step_frame,linker_bp_coarse_index_list)
        
        # todo, rewrite in pythonic way
        if misc_beads is None:
            all_beads=np.vstack((dna_beads,ncp_beads))
        else:
            all_beads=np.vstack((dna_beads,ncp_beads,misc_beads))

        force_params=get_real_space_force_mat(dna_beads,ncp_beads,misc_beads,
                                              dist_pairs=self.restrains,**energy_dict)
        if not(excluded):
            force_params['radii_sum_prod']=None
        if not(electrostatic):
            force_params['charges_multipl_prod']=None
        if not(dist_pairs):
            force_params['pair_indexes']=None
        
        linker_bp_index_list = np.hstack(linker_bp_index_list)
        linker_bp_index_list = linker_bp_index_list[linker_bp_index_list > self.entities['NCP'][0]['loc']]
        linker_bp_index_list = linker_bp_index_list[linker_bp_index_list < self.entities['NCP'][len(self.entities['NCP'])-1]['loc']]
        
        current_frame = self.bp_step_frame.copy()
        prev_e = np.finfo(float).max
        frames = []
        energies = []
        real_space = []
        total_step_bar = tqdm(total=max_steps,desc='Steps')
        #linker_pbar = tqdm(total=len(linker_bp_index_list),desc='Linker cycles',disable=mute)
        info_pbar=  tqdm(total=100,bar_format='{l_bar}{bar}{postfix}',desc='Acceptance rate',disable=mute)
        total_step=0
        accepted_steps=0
        last_accepted=0
        
        
        try:
            while total_step < max_steps and accepted_steps < target_accepted_steps:
                # resetting the generator
                np.random.seed()




                for linker_bp_index in linker_bp_index_list:
                    #linker_pbar.update(j+1-linker_pbar.n)
                    temp_frame=current_frame.copy()
                    temp_frame[linker_bp_index,6:]+=np.random.normal(scale=scale,size=[6])
                    sub_frame=get_bpstep_frame(temp_frame,movable_steps)
                    bend_e=_calc_bend_energy(sub_frame,force_matrix,average_bpstep_frame)

                    if accepted_steps%rebuild_index == 0:
                        dna_beads,ncp_beads,misc_beads,cur_ref_frames = self.get_all_beads_on_fiber(temp_frame,
                                                                                 linker_bp_coarse_index_list)
                    else:
                        dna_beads,ncp_beads,misc_beads,cur_ref_frames = self.get_all_beads_on_fiber_no_rebuild(ncp_beads,misc_beads,prev_ref_frames,
                                                         temp_frame[linker_bp_index],linker_bp_index,linker_bp_coarse_index_list)
                        dna_beads1,ncp_beads1,misc_beads1,cur_ref_frames1 = self.get_all_beads_on_fiber(temp_frame,
                                                                             linker_bp_coarse_index_list)
                        print(np.average(np.linalg.norm(cur_ref_frames[:,3,:3]-cur_ref_frames1[:,3,:3])),linker_bp_index)

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
                            frames.append(current_frame)
                            components['bend']=bend_e
                            components['entities']=copy.deepcopy(self.entities)
                            components['linker_mask']=copy.deepcopy(np.sort(np.hstack(linker_bp_index_list)))
                            energies.append(components)
                            if save_beads:
                                real_space.append({'dna':dna_beads,'ncp':ncp_beads,'misc':misc_beads})
                    info_pbar.set_postfix(E=f'{prev_e:.2f}, Acc.st {accepted_steps}')
                    info_pbar.set_description(f'Acceptance rate %')
                    acceptance_rate=100*accepted_steps/total_step
                    info_pbar.update(acceptance_rate-info_pbar.n)
                    
        
        except KeyboardInterrupt:
            print('interrupted!')
        if accepted_steps >= target_accepted_steps:
            print('target accepted steps reached')
        print('accepted_steps',accepted_steps)
        return(frames,energies)
    
    
    
    def start_mc_by_linker(self,bending=True,excluded=True,
                   electrostatic=False,dist_pairs=True,energy_dict={},
                           KT=1,sigma_transl=1,sigma_rot=1,
                           max_trials_per_linker=10,n_macrocycles=1,
                           single_step_var=False,single_var_dev=False,
                           mute=False,save_every=1,save_beads=False,ncp_remodel_dict=None):
        '''
        TODO 
        rebuild after init
        SAVE METAINFO OF START INCLUDING THE DATE
        show graphs with energy and stuff
        SAVE AS MDA TRAJ
        WRITE DIRECTLY INTO OBJECT TO USE LATTER
        ncp_remodel_dict : {macrocycle_step:50,target_dist:160,step_size:1,compensate_twist_steps:None}
        режимы:
        линкеры по очереди
        случайные линкеры
        нуклеотиды по очереди - тут возможен чит, когда не надо пересчитывать все, а поворачивать все биды (должно быть быстрее, чем пересчитывать все с нуля)
         - случайные 6 параметров
         - случайный 1 параметр
         случайные нуклеотиды
         - случайные 6 параметров
         - случайный 1 параметр
        energy_dict={
        'dna_r':5,'ncp_r':60,'misc_r':5,
        'dna_eps':0.5,'ncp_eps':1,'misc_eps':0,
        'dna_q':0,'ncp_q':0,'misc_q':0,
        'K_excl':1,'K_elec':1,'K_dist':1,'K_bend':1}
        https://en.wikibooks.org/wiki/Molecular_Simulation/Monte_Carlo_Methods
        '''
        scale=[*[sigma_transl]*3,*[sigma_rot]*3]
        #setting up DNA bp space
        if not 'K_bend' in energy_dict:
            energy_dict['K_bend']=1
        K_bend=energy_dict['K_bend']
        AVERAGE,FORCE_CONST,DISP=get_consts_olson_98()
        movable_steps=np.argwhere(self.linker_mask).flatten()[1:]-1

        force_matrix=get_force_matrix(self.pairs,movable_steps,FORCE_CONST)
        average_bpstep_frame=get_force_matrix(self.pairs,movable_steps,AVERAGE)        
        
        # setting up real space
        linker_bp_index_list,linker_bp_coarse_index_list=get_linkers_bp_indexes(self.linker_mask)
        dna_beads,ncp_beads,misc_beads,cur_ref_frames=self.get_all_beads_on_fiber(self.bp_step_frame,linker_bp_coarse_index_list)
        if misc_beads is None:
            all_beads=np.vstack((dna_beads,ncp_beads))
        else:
            all_beads=np.vstack((dna_beads,ncp_beads,misc_beads))

        force_params=get_real_space_force_mat(dna_beads,ncp_beads,misc_beads,
                                              dist_pairs=self.restrains,**energy_dict)
        if not(excluded):
            force_params['radii_sum_prod']=None
        if not(electrostatic):
            force_params['charges_multipl_prod']=None
        if not(dist_pairs):
            force_params['pair_indexes']=None
        
        if single_step_var:
            linker_bp_index_list=np.hstack(linker_bp_index_list).reshape(-1,1)
            linker_bp_index_list = linker_bp_index_list[linker_bp_index_list[:,0] != 0]
        start_linker_bp_index_list = linker_bp_index_list.copy()

        current_frame=self.bp_step_frame.copy()
        prev_e=np.finfo(float).max
        frames=[]
        energies=[]
        real_space=[]
        macro_pbar = tqdm(np.arange(n_macrocycles),desc='Macro cycles')
        #linker_pbar = tqdm(total=len(linker_bp_index_list),desc='Linker cycles',disable=mute)
        info_pbar=  tqdm(total=100,bar_format='{l_bar}{bar}{postfix}',desc='Acceptance rate',disable=mute)
        total_step=0
        accepted_steps=0
        last_accepted=0
        ncp_moves =0
        ncp_movefails =0
        if not (ncp_remodel_dict is None):
            ncp_move_tries = ncp_remodel_dict.get('retry_num',0)
        current_move_try = 0
        start_KT = copy.deepcopy(KT)
        
        try:
            for i in macro_pbar:
                # resetting the generator
                np.random.seed()




                for j,linker_bp_indexes in enumerate(linker_bp_index_list):
                    #linker_pbar.update(j+1-linker_pbar.n)
                    temp_frame=current_frame.copy()
                    trial=0
                    while trial < max_trials_per_linker:
                        trial+=1
                        if single_var_dev:
                            var_id=np.random.randint(0, high=6)
                            temp_frame[linker_bp_indexes,6+var_id]+=np.random.normal(scale=scale[var_id])
                        else:
                            temp_frame[linker_bp_indexes,6:]+=np.random.normal(scale=scale,size=[len(linker_bp_indexes),6])
                        sub_frame=get_bpstep_frame(temp_frame,movable_steps)
                        bend_e=_calc_bend_energy(sub_frame,force_matrix,average_bpstep_frame)

                        dna_beads,ncp_beads,misc_beads,cur_ref_frames = self.get_all_beads_on_fiber(temp_frame,
                                                                                     linker_bp_coarse_index_list)
                        if misc_beads is None:
                            all_beads=np.vstack((dna_beads,ncp_beads))
                        else:
                            all_beads=np.vstack((dna_beads,ncp_beads,misc_beads))
                        real_e,components=calc_real_space_total_energy(all_beads,**force_params,**energy_dict)
                        total_e=real_e+K_bend*bend_e
                        Del_E=total_e-prev_e

                        r = np.random.uniform(0,1)
                        total_step+=1
                        if Del_E < 0 or (not(np.isinf(np.exp(Del_E))) and (r  <= np.exp(-Del_E/KT))):
                            prev_e=total_e
                            accepted_steps+=1
                            #print(prev_e)                       

                            current_frame=temp_frame.copy()
                            if (accepted_steps% save_every) == 0:
                                frames.append(current_frame)
                                components['bend']=bend_e
                                components['macro_cycle']=i
                                components['entities']=copy.deepcopy(self.entities)
                                
                                components['linker_mask']=copy.deepcopy(np.sort(linker_bp_index_list))
                                energies.append(components)
                                if save_beads:
                                    real_space.append({'dna':dna_beads,'ncp':ncp_beads,'misc':misc_beads})
                            break
                        info_pbar.set_postfix(E=f'{prev_e:.2f}, Acc.st {len(energies)}, Acc.m {ncp_moves-ncp_movefails}, failed.m {ncp_movefails}')
                        info_pbar.set_description(f'Acceptance rate %')
                        acceptance_rate=100*accepted_steps/total_step
                        info_pbar.update(acceptance_rate-info_pbar.n)
        
        except KeyboardInterrupt:
            print('interrupted!')
        print('NCP_moves:',ncp_moves, ncp_movefails)
        print('accepted_steps',accepted_steps)
        return(frames,energies)
    
    
    
#------------------------------------------------------------------------------------    
    
    def start_mc_by_linker_2(self,bending=True,excluded=True,
                   electrostatic=False,dist_pairs=True,energy_dict={},
                           KT=1,sigma_transl=1,sigma_rot=1,n_macrocycles=1,
                           single_var_dev=False, mute=False,save_frames_every=1,
                           ncp_remodel_dict=None):
        '''

        https://en.wikibooks.org/wiki/Molecular_Simulation/Monte_Carlo_Methods
        '''
        scale=[*[sigma_transl]*3,*[sigma_rot]*3]
        AVERAGE,FORCE_CONST,DISP=get_consts_olson_98()
        if not 'K_bend' in energy_dict:
            energy_dict['K_bend']=1
        K_bend=energy_dict['K_bend']
        
        
        def update_NCPS_linkers_and_force_matrices(self, new_entities_dict, old_entities_dict, old_bp_step_frame):
            '''
            This function updates pairs frame by moving
            '''
            
            bp_step_frame = old_bp_step_frame.copy()
            linker_mask = np.ones(old_bp_step_frame.shape[0]).astype(bool)
            
            for ncp_num,old_ncp in old_entities_dict['NCP'].items():
                new_ncp = new_entities_dict['NCP'][ncp_num]
                
                old_dyad_loc = old_ncp['loc']
                old_entry_unwrap = old_ncp['entry_unwrap']
                old_exit_unwrap = old_ncp['exit_unwrap']
                old_left_dyad_offset = old_ncp['left_dyad_offset']
                old_right_dyad_offset = old_ncp['right_dyad_offset']
                
                new_dyad_loc = new_ncp['loc']
                new_entry_unwrap = new_ncp['entry_unwrap']
                new_exit_unwrap = new_ncp['exit_unwrap']
                new_left_dyad_offset = new_ncp['left_dyad_offset']
                new_right_dyad_offset = new_ncp['right_dyad_offset']
                #updating linker mask
                linker_mask  [new_dyad_loc + new_left_dyad_offset + new_entry_unwrap : new_dyad_loc + new_right_dyad_offset - new_exit_unwrap + 1] = False
                #updating bp_ref frame according to new locations
                bp_step_frame[new_dyad_loc + new_left_dyad_offset + new_entry_unwrap :
                              new_dyad_loc + new_right_dyad_offset - new_exit_unwrap + 1, 6:] = old_bp_step_frame[old_dyad_loc + old_left_dyad_offset + old_entry_unwrap :
                                                                                                                 old_dyad_loc + old_right_dyad_offset - old_exit_unwrap + 1,6:]

                
            
            movable_steps=np.argwhere(linker_mask).flatten()[1:]-1

            force_matrix=get_force_matrix(self.pairs,movable_steps,FORCE_CONST)
            average_bpstep_frame=get_force_matrix(self.pairs,movable_steps,AVERAGE)        
        
            # setting up real space
            linker_bp_index_list,linker_bp_coarse_index_list=get_linkers_bp_indexes(linker_mask)
            dna_beads,ncp_beads,misc_beads=self.get_all_beads_on_fiber(bp_step_frame,linker_bp_coarse_index_list,
                                                                       linker_mask = linker_mask, entities = new_entities_dict)
            if misc_beads is None:
                all_beads=np.vstack((dna_beads,ncp_beads))
            else:
                all_beads=np.vstack((dna_beads,ncp_beads,misc_beads))
            force_params=get_real_space_force_mat(dna_beads,ncp_beads,misc_beads,
                                                  dist_pairs=self.restrains,**energy_dict)
            if not(excluded):
                force_params['radii_sum_prod']=None
            if not(electrostatic):
                force_params['charges_multipl_prod']=None
            if not(dist_pairs):
                force_params['pair_indexes']=None
            linker_bp_index_list=np.hstack(linker_bp_index_list).reshape(-1,1)
            
            return linker_bp_index_list,linker_bp_coarse_index_list, force_params, movable_steps, force_matrix, average_bpstep_frame, bp_step_frame
        
        linker_bp_index_list,linker_bp_coarse_index_list, force_params, \
        movable_steps, force_matrix, average_bpstep_frame, current_frame = update_NCPS_linkers_and_force_matrices(self, self.entities, self.entities, self.bp_step_frame)
        
        current_entities = copy.deepcopy(self.entities)
        
        prev_e = np.finfo(float).max
        
        frames=[]
        energies=[]
        real_space=[]
        macro_pbar = tqdm(np.arange(n_macrocycles),desc='All linkers')
        info_pbar  = tqdm(total=100,bar_format='{l_bar}{bar}{postfix}',desc='Acceptance rate',disable=mute)

        total_step     = 0
        accepted_steps = 0
        remodel_flag = False
        remodel_attempt = 0
        
        succ_remodel_steps = 0
        failed_remodel_steps = 0
        
        start_KT = copy.deepcopy(KT)
        try:
            for i in macro_pbar:
                # resetting the generator
                np.random.seed()

                # optimisation

                for j,linker_bp_indexes in enumerate(linker_bp_index_list):

                    #linker_pbar.update(j+1-linker_pbar.n)
                    temp_frame=current_frame.copy()

                    if single_var_dev:
                        var_id=np.random.randint(0, high=6)
                        temp_frame[linker_bp_indexes,6+var_id]+=np.random.normal(scale=scale[var_id])
                    else:
                        temp_frame[linker_bp_indexes,6:]+=np.random.normal(scale=scale,size=[len(linker_bp_indexes),6])
                    sub_frame=get_bpstep_frame(temp_frame,movable_steps)
                    bend_e=_calc_bend_energy(sub_frame,force_matrix,average_bpstep_frame)

                    dna_beads,ncp_beads,misc_beads = self.get_all_beads_on_fiber(temp_frame,
                                                                                 linker_bp_coarse_index_list)
                    if misc_beads is None:
                        all_beads=np.vstack((dna_beads,ncp_beads))
                    else:
                        all_beads=np.vstack((dna_beads,ncp_beads,misc_beads))
                    real_e,components=calc_real_space_total_energy(all_beads,**force_params,**energy_dict)
                    #total_e = real_e / len(all_beads) + K_bend * bend_e
                    total_e = real_e + K_bend*bend_e
                    Del_E = total_e - prev_e

                    r = np.random.uniform(0,1)
                    total_step+=1
                    if Del_E < 0 or (not(np.isinf(np.exp(Del_E))) and (r  <= np.exp(-Del_E/KT))):
                        prev_e=total_e
                        accepted_steps+=1
                        #print(prev_e)                       

                        current_frame=temp_frame.copy()
                        if remodel_flag: succ_remodel_steps+=1
                        remodel_flag = False
                        if (accepted_steps% save_frames_every) == 0:
                            frames.append(current_frame)
                            components['bend']=bend_e
                            components['macro_cycle']=i
                            components['entities']=copy.deepcopy(current_entities)
                            components['linker_mask']=copy.deepcopy(np.sort(linker_bp_index_list))
                            energies.append(components)

                        info_pbar.set_postfix(E=f'{total_e:.2f},B={bend_e:.2f},V={components["vdv"]:.2f} Acc.st {len(energies)},{succ_remodel_steps},{failed_remodel_steps}')#', Acc.m {ncp_moves-ncp_movefails}, failed.m {ncp_movefails}')
                        info_pbar.set_description(f'Acceptance rate %')
                        acceptance_rate=100*accepted_steps/total_step
                        info_pbar.update(acceptance_rate-info_pbar.n)                      
        

        #print('NCP_moves:',ncp_moves, ncp_movefails)
        except KeyboardInterrupt:
            print('interrupted!')
        #print('accepted_steps',accepted_steps)
        return(frames,energies)
    
    def to_mda_traj(self,frame_traj=None,n_threads=20,entities_list=None):
        model=self
        linker_bp_index_list,linker_bp_coarse_index_list=get_linkers_bp_indexes(model.linker_mask)
        if frame_traj is None:
            beads=[model_recalc_worker((model.bp_step_frame,model,linker_bp_coarse_index_list))]
        else:
            if entities_list is None:
                iterable = [(bp_step_frame,model,linker_bp_coarse_index_list) for bp_step_frame in frame_traj]
            else:
                assert len(entities_list)==len(frame_traj)
                iterable = [(bp_step_frame,model,linker_bp_coarse_index_list,entities) for bp_step_frame,entities in zip(frame_traj,entities_list)]
            if n_threads==1:
                beads = [model_recalc_worker(iteration) for iteration in tqdm(iterable)]
            else:
                with Pool(n_threads) as p:
                    iterator=p.imap(model_recalc_worker, iterable)
                    beads=list(tqdm(iterator,total=len(frame_traj)))
    #     return beads    
        resnames = list(itertools.chain(*[[name]*N for name,N in zip(['DNA','NCP','MSC'],map(len,beads[0]))]))
        resids = list(itertools.chain(*[np.arange(N) for name,N in zip([0,1,2],map(len,beads[0]))]))
        segids = list(itertools.chain(*[[name]*N for name,N in zip([0,1,2],map(len,beads[0]))]))
        masses = list(itertools.chain(*[[name]*N for name,N in zip([700,5000,1],map(len,beads[0]))]))
        if beads[0][2]:
            coords = np.array(list(map(np.vstack,beads))) # frame atom coord
        else:
            coords = np.array(list(map(lambda bead: np.vstack(bead[:2]),beads)))
        n_frames, n_atoms,_ = coords.shape

        u = mda.Universe.empty(n_atoms,n_residues=n_atoms,n_segments=3,
                           atom_resindex=np.arange(n_atoms),
                           residue_segindex=segids,
                           trajectory=True)
        u.add_TopologyAttr('name',resnames)
        u.add_TopologyAttr('resname',resnames)
        u.add_TopologyAttr('resid',resids)
        u.add_TopologyAttr('segid',['D','N','M'])
        u.add_TopologyAttr('mass',masses)
        #altLocs icodes occupancies tempfactors
        u.load_new(coords/10, format=MemoryReader)
        alignment = align.AlignTraj(u, u,in_memory=True)
        alignment.run()        

        return(u)


        


def get_linkers_bp_indexes(linker_mask,linker_coarse_step=5,linker_min_offset=3):
    linker_bp_index_list=[]
    linker_bp_coarse_index_list=[]
    linker_bp_index = np.argwhere(linker_mask).flatten()
    last_id=0
    for cur_id in np.argwhere((linker_bp_index[1:]-linker_bp_index[:-1]!=1)).flatten():
        linker_bp_index_list.append(linker_bp_index[last_id:cur_id+1])
        last_id=cur_id+1
    if last_id!=len(linker_bp_index):
        linker_bp_index_list.append(linker_bp_index[last_id:])

    for linker in linker_bp_index_list:
        sel=linker[linker_min_offset:-linker_min_offset+1]
        linker_bp_coarse_index_list.append(sel[(len(sel)//2)%linker_coarse_step::linker_coarse_step])
        #linker_bp_coarse_index_list.append(sel[::linker_coarse_step])
    linker_bp_coarse_index_list=np.concatenate(linker_bp_coarse_index_list)

    return(linker_bp_index_list,linker_bp_coarse_index_list)
    
import pickle
from os.path import exists
def load_ncp(pdbid):
    logging.info(f'Loading {pdbid}')
    if exists('.'+pdbid):
        logging.info(f'Found structure cached in .{pdbid}')
        with open('.'+pdbid,'rb') as fp:
            (R2,of_vec),ncp_sequence,bp_step_frame = pickle.load(fp)
            return((R2,of_vec),ncp_sequence,bp_step_frame)
    try:
        h=io.StringIO(requests.get(f'https://files.rcsb.org/download/{pdbid.lower()}.pdb').content.decode("utf-8"))
    except:
        logging.warning(f'Failed to load file {pdbid}')
        return
    p=pynucl.nuclstr(h,format='PDB',auto_detect_entities=True)
    a=pynucl.a_DNA(p,num_threads=1)
    bp_step_frame=a.df[['BPnum_dyad','Shear','Stretch','Stagger','Buckle','Prop-Tw','Opening','Shift','Slide','Rise','Tilt','Roll','Twist']]
    bp_step_frame.set_index('BPnum_dyad',inplace=True)
    ncp_sequence=''
    for index,row in a.df.set_index('BPnum_dyad')[['segid','resid']].iterrows():
        sel=p.u.select_atoms(f"segid {row['segid']} and resnum {row['resid']}")
        ncp_sequence+=residues_3to1[sel.residues.resnames[0]]
        
    # this code needs refactoring and assumes that ncp has at least 40 bp both sides    
    ref_frames=rebuild_by_full_par_frame_numba(bp_step_frame.to_numpy())
    b1= bp_step_frame.index.get_loc(-39)
    b2= bp_step_frame.index.get_loc( 39)
    t = bp_step_frame.index.get_loc( 0 )
    o1 = ref_frames[t,3,:3]
    o2 = 0.5*(ref_frames[t,3,:3]+ref_frames[[b1,b2],3,:3].mean(0))
    ref_mat=ref_frames[t,:3,:3]
    obj_mat=np.identity(3)
    R2,of_vec=get_rotation_and_offset_ref_frame_to_obj(ref_mat,o1,obj_mat,o2)
    logging.info(f'Saving cached data to .{pdbid}')
    with open('.'+pdbid,'wb') as fp:
        pickle.dump(((R2,of_vec),ncp_sequence,bp_step_frame),fp)
    
    return((R2,of_vec),ncp_sequence,bp_step_frame)
        
        
def model_recalc_worker(iter_tuple):
    if len(iter_tuple)==3:
        
        bp_step_frame,model,linker_bp_coarse_index_list = iter_tuple
    else:
        bp_step_frame,model,linker_bp_coarse_index_list,entities = iter_tuple
        model.entities = entities
    cur_ref_frames=rebuild_by_full_par_frame_numba(bp_step_frame)
    dna_beads,ncp_beads,misc_beads=model.get_all_beads_on_fiber(bp_step_frame,linker_bp_coarse_index_list)
    return(cur_ref_frames[:,3,:3],ncp_beads,[] if misc_beads is None else misc_beads)
        