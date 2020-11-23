import requests
import io
import pynucl
from pynucl.seq_utils import residues_3to1
from pynamod.energy_constants import * 
from pynamod.visual_ngl import show_ref_frames
from pynamod.non_DNA_geometry import get_obj_orientation_and_location,get_rotation_and_offset_ref_frame_to_obj
from pynamod.bp_step_geometry import rebuild_by_full_par_frame_numba
from pynamod.energy_funcs import *
from pynamod.energy_funcs import _calc_bend_energy
from pynamod.utils import get_movable_steps
from tqdm.auto import tqdm,trange

import logging

from pynamod.utils import W_C_pair
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
            
            
        if isinstance(sequence,str) and (set('ATGC') == set(sequence.upper())):
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
                
        if (linker_lengths==None ) and (len(self.ncp_pdb_ids)==len(ncp_dyad_locations)):
            self.place_ncps_on_fiber_by_dyad_pos(self.ncp_pdb_ids,ncp_dyad_locations)
        elif (linker_lengths==None ) and (len(self.ncp_pdb_ids)!=len(ncp_dyad_locations)):
            self.place_ncps_on_fiber_by_dyad_pos(self.ncp_pdb_ids,ncp_dyad_locations)
            logging.warning('ncp_pdb_ids and ncp_dyad_locations does not match, or no linker lengths provided')
        elif isinstance(linker_lengths,int) or (len(linker_lengths) == (len(self.ncp_pdb_ids)+1)):
            logging.warning('ncp_dyad_locations ignored as linker_sequence or linker_lengths provided')
            self.place_ncps_on_fiber_by_linker_lengths(self.ncp_pdb_ids,linker_lengths)
            
        else:
            logging.warning('ncp_pdb_ids and ncp_dyad_locations does not match, or no linker lengths provided')
        
    def place_ncps_on_fiber_by_linker_lengths(self,ncp_pdb_ids,linker_lengths):
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
        self.place_ncps_on_fiber_by_dyad_pos(self.ncp_pdb_ids,ncp_dyad_locations)
        
    def place_ncps_on_fiber_by_dyad_pos(self,ncp_pdb_ids,ncp_dyad_locations):
        self.entities['NCP']={}
        self.ncp_dyad_locations=ncp_dyad_locations
        self.ncp_pdb_ids=ncp_pdb_ids=[ncp_pdb_id.lower().strip() for ncp_pdb_id in ncp_pdb_ids]
        self.bp_step_frame=self.initial_bp_step_frame.copy()
        self.linker_mask=np.ones(self.bp_step_frame.shape[0]).astype(bool)
        for i,(ncp_id,dyad_loc) in enumerate(zip(ncp_pdb_ids,ncp_dyad_locations)):
            if not ncp_id in self.ncp_bp_step_frames.keys():
                self.ncp_offsets[ncp_id], self.ncp_sequences[ncp_id],self.ncp_bp_step_frames[ncp_id]=load_ncp(ncp_id)
            offset0=self.ncp_bp_step_frames[ncp_id].index[0]
            offset1=self.ncp_bp_step_frames[ncp_id].index[-1]
            ncp_frame_np=self.ncp_bp_step_frames[ncp_id].to_numpy()
            if ((dyad_loc+offset0) >= 0) and ((dyad_loc+offset1) < len(self.sequence)):
                self.bp_step_frame[dyad_loc+offset0:dyad_loc+offset1+1,:6]=ncp_frame_np[:,:6]
                self.bp_step_frame[dyad_loc+offset0+1:dyad_loc+offset1+1,6:]=ncp_frame_np[1:,6:]
                self.linker_mask[dyad_loc+offset0:dyad_loc+offset1+1]=False
                self.entities['NCP'][i]={'name':f'ncp_i',
                                         'loc' :dyad_loc,
                                         'kind':'NCP',
                                         'type':ncp_id,
                                         'v_offset':self.ncp_offsets[ncp_id][1],
                                         'r_mat'   :self.ncp_offsets[ncp_id][0],
                                         'R':45}
                
                
            else:
                logging.warning(f' NCP #{i} {ncp_id} located on {dyad_loc} is out of sequence bounds, skipped')
                
    def add_restraint(self,a_num,b_num,dist,a_kind='ncp',b_kind='ncp'):
        restraint={'a':[a_kind,a_num],'b':[b_kind,b_num],'dist':dist}
        if not (restraint in self.restrains):
            self.restrains.append({'a':[a_kind,a_num],'b':[b_kind,b_num],'dist':dist})
    
    def view_fiber(self,bp_step_frame=None,spheres=True,arrows=False,diamonds=False,boxes=False,
                   show_model=False):
        bp_step_frame = self.bp_step_frame if bp_step_frame is None else bp_step_frame
        self.cur_ref_frames=rebuild_by_full_par_frame_numba(bp_step_frame)
        view=show_ref_frames(self.cur_ref_frames,spheres=spheres,arrows=arrows,diamonds=diamonds,boxes=boxes,
                            bp_colors_mask=self.linker_mask)
        self.ncp_beads=np.zeros((len(self.entities['NCP']),3))
        cylynders=np.zeros((len(self.entities['NCP']),2,3))
        
        linker_bp_index_list,linker_bp_coarse_index_list=get_linkers_bp_indexes(self.linker_mask)
        dna_beads,self.ncp_beads,misc_beads=self.get_all_beads_on_fiber(bp_step_frame,linker_bp_coarse_index_list)
        print(dna_beads.shape)   
    
        view.shape.add_buffer('sphere',position=self.ncp_beads.flatten().tolist(),
                                  color=[1,0,0]*self.ncp_beads.shape[0],
                                  radius=  [40]*self.ncp_beads.shape[0])
        if show_model:
            view.shape.add_buffer('sphere',position=dna_beads.flatten().tolist(),
                                  color=[0,1,0]*dna_beads.shape[0],
                                  radius=  [10]*dna_beads.shape[0])
            if not (misc_beads is None):
                view.shape.add_buffer('sphere',position=misc_beads.flatten().tolist(),
                                  color=[0,1,1]*misc_beads.shape[0],
                                  radius=  [20]*misc_beads.shape[0])
        
#         elif ncp_repr=='cylinder':
#                     view.shape.add_buffer('cylinder',
#                               position1=cylynders[:,0].flatten().tolist(),
#                               position2=cylynders[:,1].flatten().tolist(),
#                               color=[1,0,0]*(self.ncp_beads.shape[0]),radius=[30]*self.ncp_beads.shape[0])
        return(view)
    
    
    
    def get_all_beads_on_fiber(self,bp_step_frame, linker_bp_coarse_index_list=None):
        cur_ref_frames=rebuild_by_full_par_frame_numba(bp_step_frame)
        
        ### DNA BEADS ###
        if linker_bp_coarse_index_list is None:
            linker_bp_coarse_index_list=self.linker_mask
        dna_beads=cur_ref_frames[linker_bp_coarse_index_list,3,:3]
        
        ### NCP BEADS ###
        ncp_beads=np.zeros((len(self.entities['NCP']),3))           
        for num,NCP in self.entities['NCP'].items():
            ref_mat=cur_ref_frames[NCP['loc'],:3,:3]
            o1=cur_ref_frames[NCP['loc'],3,:3]
            R2,of_vec=NCP['r_mat'],NCP['v_offset']
            calcR2,calcO2=get_obj_orientation_and_location(ref_mat,o1,R2,of_vec)
            ncp_beads[num]=calcO2
         ### MISC BEADS ###
        if len(self.entities['misc'])!=0:
            misc_beads=np.zeros((len(self.entities['misc']),3))           
            for num,NCP in self.entities['misc'].items():
                ref_mat=cur_ref_frames[NCP['loc'],:3,:3]
                o1=cur_ref_frames[NCP['loc'],3,:3]
                R2,of_vec=NCP['r_mat'],NCP['v_offset']
                calcR2,calcO2=get_obj_orientation_and_location(ref_mat,o1,R2,of_vec)
                misc_beads[num]=calcO2
        else:
            misc_beads=None
        return(dna_beads,ncp_beads,misc_beads)
        
        
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
        
        bend_e = energy_dict['K_bend'] * get_bend_energy(self.bp_step_frame,self.pairs,movable_steps,FORCE_CONST,AVERAGE)
        # real space
        
        linker_bp_index_list,linker_bp_coarse_index_list=get_linkers_bp_indexes(self.linker_mask)
        dna_beads,ncp_beads,misc_beads=self.get_all_beads_on_fiber(self.bp_step_frame,linker_bp_coarse_index_list)
        if misc_beads is None:
            all_beads=np.vstack((dna_beads,ncp_beads))
        else:
            all_beads=np.vstack((dna_beads,ncp_beads,misc_beads))
        if dist_pairs:
            force_params = get_real_space_force_mat(dna_beads,ncp_beads,misc_beads,dist_pairs=self.restrains,**energy_dict)
        else:
            force_params=get_real_space_force_mat(dna_beads,ncp_beads,misc_beads,**energy_dict)
            
        real_space_e=calc_real_space_total_energy(all_beads,**force_params,**energy_dict)
        real_space_e[1]['bend']=bend_e
        return(real_space_e[0]+bend_e,real_space_e[1])
    
    
    def start_mc_by_linker(self,bending=True,excluded=True,
                   electrostatic=False,dist_pairs=True,energy_dict={},
                           KT=1,sigma_transl=1,sigma_rot=1,
                           max_trials_per_linker=10,n_macrocycles=1):
        '''
        TODO SAVE METAINFO OF START INCLUDING THE DATE
        show graphs with energy and stuff
        SAVE AS MDA TRAJ
        WRITE DIRECTLY INTO OBJECT TO USE LATTER
        VIEW restraints
        ребилд нужен бы еще
        energy_dict={
        'dna_r':5,'ncp_r':60,'misc_r':5,
        'dna_eps':0.5,'ncp_eps':1,'misc_eps':0,
        'dna_q':0,'ncp_q':0,'misc_q':0,
        'K_excl':1,'K_elec':1,'K_dist':1,'K_bend':1}
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
        dna_beads,ncp_beads,misc_beads=self.get_all_beads_on_fiber(self.bp_step_frame,linker_bp_coarse_index_list)
        if misc_beads is None:
            all_beads=np.vstack((dna_beads,ncp_beads))
        else:
            all_beads=np.vstack((dna_beads,ncp_beads,misc_beads))

        force_params=get_real_space_force_mat(dna_beads,ncp_beads,misc_beads,dist_pairs=self.restrains,**energy_dict)
        if not(excluded):
            force_params['radii_sum_prod']=None
        if not(electrostatic):
            force_params['charges_multipl_prod']=None
        if not(dist_pairs):
            force_params['pair_indexes']=None
        
        current_frame=self.bp_step_frame.copy()
        prev_e=np.finfo(float).max
        frames=[]
        energies=[]
        macro_pbar = trange(n_macrocycles,desc='Macro cycles')
        linker_pbar = tqdm(total=len(linker_bp_index_list),desc='Linker cycles')
        info_pbar=  tqdm(total=100,bar_format='{l_bar}{bar}{postfix}',desc='Acceptance rate')
        total_step=0
        for i in macro_pbar:
            # resetting the generator
            #np.random.seed()
            for j,linker_bp_indexes in enumerate(linker_bp_index_list):
                linker_pbar.update(j+1-linker_pbar.n)
                temp_frame=current_frame.copy()
                trial=0
                while trial < max_trials_per_linker:
                    trial+=1
                    temp_frame[linker_bp_indexes,6:]+=np.random.normal(scale=scale,size=[len(linker_bp_indexes),6])
                    sub_frame=get_bpstep_frame(temp_frame,movable_steps)
                    bend_e=_calc_bend_energy(sub_frame,force_matrix,average_bpstep_frame)

                    dna_beads,ncp_beads,misc_beads = self.get_all_beads_on_fiber(temp_frame,linker_bp_coarse_index_list)
                    if misc_beads is None:
                        all_beads=np.vstack((dna_beads,ncp_beads))
                    else:
                        all_beads=np.vstack((dna_beads,ncp_beads,misc_beads))
                    real_e,components=calc_real_space_total_energy(all_beads,**force_params)
                    total_e=real_e+K_bend*bend_e
                    Del_E=total_e-prev_e
                    
                    r = np.random.uniform(0,1)
                    total_step+=1
                    if Del_E < 0 or (not(np.isinf(np.exp(Del_E))) and (r  < np.exp(-Del_E/KT))):
                        prev_e=total_e
                        #print(prev_e)                       
                        
                        current_frame=temp_frame.copy()
                        frames.append(current_frame)
                        components['bend']=bend_e
                        energies.append(components)
                        info_pbar.set_postfix(E=f'{prev_e:.2f}, Accepted steps {len(energies)}')
                        info_pbar.set_description(f'Acceptance rate %')
                        acceptance_rate=100*len(energies)/total_step
                        info_pbar.update(acceptance_rate-info_pbar.n)
                        break
        return(frames,energies)
        


def get_linkers_bp_indexes(linker_mask,linker_coarse_step=5,linker_min_offset=2):
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
        sel=linker[linker_min_offset:-linker_min_offset]
        linker_bp_coarse_index_list.append(sel[(len(sel)//2)%linker_coarse_step::linker_coarse_step])
    linker_bp_coarse_index_list=np.concatenate(linker_bp_coarse_index_list)

    return(linker_bp_index_list,linker_bp_coarse_index_list)
    

def load_ncp(pdbid):
    logging.info(f'Loading {pdbid}')
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
    b1=bp_step_frame.index.get_loc(-39)
    b2=bp_step_frame.index.get_loc( 39)
    t =bp_step_frame.index.get_loc( 0 )
    o1 = ref_frames[t,3,:3]
    o2 = 0.5*(ref_frames[t,3,:3]+ref_frames[[b1,b2],3,:3].mean(0))
    ref_mat=ref_frames[t,:3,:3]
    obj_mat=np.identity(3)
    R2,of_vec=get_rotation_and_offset_ref_frame_to_obj(ref_mat,o1,obj_mat,o2)
    
    return((R2,of_vec),ncp_sequence,bp_step_frame)
        
        
        