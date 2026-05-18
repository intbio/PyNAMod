<<<<<<< HEAD
import pandas as pd
import numpy as np
import torch
import io
from Bio.Seq import Seq
from tqdm.auto import tqdm
from pynamod.geometry.trajectories import H5_Trajectory, Tensor_Trajectory
from pynamod.energy.energy_constants import get_consts_olson_98, BDNA_step
from pynamod.geometry.geometrical_parameters import Geometrical_Parameters
from pynamod.atomic_analysis.nucleotides_parser import get_all_nucleotides, Nucleotide, get_base_ref_frame, get_base_u
from pynamod.atomic_analysis.pairs_parser import get_pairs, Base_Pair
from pynamod.atomic_analysis.structures_storage import Nucleotides_Storage, Pairs_Storage
=======
import nglview as nv
import numpy as np
import pandas as pd
import torch
from Bio.Seq import Seq
from tqdm.auto import tqdm

from pynamod.atomic_analysis.nucleotides_parser import (Nucleotide,
                                                        get_all_nucleotides,
                                                        get_base_ref_frame)
from pynamod.atomic_analysis.pairs_parser import Base_Pair, get_pairs
from pynamod.atomic_analysis.structures_storage import (Nucleotides_Storage,
                                                        Pairs_Storage)
from pynamod.energy.energy_constants import BDNA_step, get_consts_olson_98
from pynamod.geometry.geometrical_parameters import Geometrical_Parameters
>>>>>>> 4c2722a3fcadace9ef261cd86d95e432a14f8376
from pynamod.geometry.tensor_subclasses import mod_Tensor
from pynamod.geometry.trajectories import H5_Trajectory, Tensor_Trajectory



class DNA_Structure:
    '''Class that stores information about DNA, pairs in it. Geometrical parameters in this class are links to the other class (All_Coords) that manages them.

        Main attributes of this class are:

<<<<<<< HEAD
        - **pairs_list** - list of Pair class objects. 
=======
        - **pairs_list** - list of Pair class objects.
>>>>>>> 4c2722a3fcadace9ef261cd86d95e432a14f8376
        - **origins**, **ref_frames**, **step_params** - properties that return tensors from All_Coords class with parameters for this DNA structure.'''

    def __init__(self, **kwards):
        self.pairs_list = []
<<<<<<< HEAD
        self.movable_steps = torch.empty(0,dtype=bool)
        for name,value in kwards.items():
            setattr(self, name, value)

    def build_from_u(self, leading_strands,pairs_in_structure = None,traj_len=1,
                     sel='(type C or type O or type N) and not protein',overwrite_existing_dna=False,movable=False,use_full_nucleotide=False):
=======
        self.movable_steps = torch.empty(0, dtype=bool)
        for name, value in kwards.items():
            setattr(self, name, value)

    def build_from_u(self, leading_strands, pairs_in_structure=None, traj_len=1,
                     sel='(type C or type O or type N) and not protein', overwrite_existing_dna=False, movable=False, use_full_nucleotide=False):
>>>>>>> 4c2722a3fcadace9ef261cd86d95e432a14f8376
        '''
        This method is called by CG_Structure.analyze_dna and runs full analysis of atomic structure.
        '''
        self.nucleotides = get_all_nucleotides(self, leading_strands, sel, use_full_nucleotide=use_full_nucleotide)

        if pairs_in_structure is not None:
            self.parse_pairs(pairs_in_structure)
            print(2)
        else:
            self.pairs_list = get_pairs(self)

        if len(self.pairs_list) == 0:
<<<<<<< HEAD
            raise RuntimeError('No pairs were found')

        self.get_geom_params(traj_len,overwrite_existing_dna)
        self._set_pair_params_list()
        self.movable_steps = torch.full((len(self.pairs_list),),movable)

    def get_geom_params(self,traj_len=1,overwrite_existing_dna=False):
        '''This method prepairs reference frames and orgins frim pairs, than creates object of All_Coords or updates existing one.'''
        ref_frames = torch.stack([pair.Rm for pair in self.pairs_list])

        origins = torch.vstack([pair.om for pair in self.pairs_list]).reshape(-1,1,3)
        if hasattr(self,'geom_params') and not overwrite_existing_dna:
            self.geom_params.get_new_params_set(ref_frames=ref_frames,origins=origins)
        else:
            self.geom_params = Geometrical_Parameters(ref_frames=ref_frames,origins=origins,traj_len = traj_len)

    def generate(self,sequence,radius=10,charge=-2,eps=0.5,movable=True):
=======
            raise TypeError('No pairs were found')

        self.get_geom_params(traj_len, overwrite_existing_dna)
        self._set_pair_params_list()
        self.movable_steps = torch.full((len(self.pairs_list),), movable)

    def get_geom_params(self, traj_len=1, overwrite_existing_dna=False):
        '''This method prepairs reference frames and orgins frim pairs, than creates object of All_Coords or updates existing one.'''
        ref_frames = torch.stack([pair.Rm for pair in self.pairs_list])

        origins = torch.vstack([pair.om for pair in self.pairs_list]).reshape(-1, 1, 3)
        if hasattr(self, 'geom_params') and not overwrite_existing_dna:
            self.geom_params.get_new_params_set(ref_frames=ref_frames, origins=origins)
        else:
            self.geom_params = Geometrical_Parameters(ref_frames=ref_frames, origins=origins, traj_len=traj_len)

    def generate(self, sequence, radius=10, charge=-2, eps=0.5, movable=True):
>>>>>>> 4c2722a3fcadace9ef261cd86d95e432a14f8376
        '''This method is called by CG_Structure.build_dna and creates linear DNA based on given sequence and BDNA step.'''
        sequence = sequence.upper()
        rev_sequence = Seq(sequence).reverse_complement()
        ln = len(sequence)

<<<<<<< HEAD
        self.nucleotides = Nucleotides_Storage(Nucleotide,None)
        self.nucleotides.restypes = list(sequence+rev_sequence[::-1])
=======
        self.nucleotides = Nucleotides_Storage(Nucleotide, None)
        self.nucleotides.restypes = list(sequence+rev_sequence)
>>>>>>> 4c2722a3fcadace9ef261cd86d95e432a14f8376
        self.nucleotides.resids = list(range(ln))*2
        self.nucleotides.segids = ['A']*ln + ['B']*ln
        self.nucleotides.leading_strands = [True]*ln + [False]*ln
        self.nucleotides.ref_frames = torch.zeros(ln, 3, 3, dtype=torch.double)
        self.nucleotides.origins = torch.zeros(ln, 1, 3, dtype=torch.double)
        self.nucleotides.s_residues = [None]*(ln*2)
        self.nucleotides.e_residues = [None]*(ln*2)
        self.nucleotides.base_pairs = [None]*(ln*2)

<<<<<<< HEAD
        # nucl_str_dict = {}
        # for base in ['A', 'T', 'G', 'C', 'U']:
        #     nucl_str_dict[base] = get_base_u(base)
        # self.nucleotides.res_atomss = []
        # for restype, ref_frame, origin in zip(self.nucleotides.restypes,self.nucleotides.ref_frames,self.nucleotides.origins):
        #     res_atoms = nucl_str_dict[restype].copy()
        #     st_positions = torch.from_numpy(res_atoms.atoms.positions).reshape(-1,1,3).to(torch.double)
        #     res_atoms.atoms.positions = torch.matmul(st_positions,ref_frame.T) + origin
        #     self.nucleotides.res_atomss.append(res_atoms)

        self.pairs_list = Pairs_Storage(Base_Pair,self.nucleotides)
=======
        self.pairs_list = Pairs_Storage(Base_Pair, self.nucleotides)
>>>>>>> 4c2722a3fcadace9ef261cd86d95e432a14f8376
        self.pairs_list.lead_nucl_inds = list(range(ln))
        self.pairs_list.lag_nucl_inds = [i+ln for i in range(ln)]
        self.pairs_list.radii = [radius]*ln
        self.pairs_list.charges = [charge]*ln
        self.pairs_list.epsilons = [eps]*ln
        pair_params = torch.zeros(2, 6, dtype=torch.double)
        pair_params[1] = torch.from_numpy(BDNA_step[:6])
        self.pairs_list.geom_paramss = [Geometrical_Parameters(local_params=pair_params.clone()) for i in range(ln)]

<<<<<<< HEAD
        step_params = torch.zeros(ln,6,dtype=torch.double)
=======
        step_params = torch.zeros(ln, 6, dtype=torch.double)
>>>>>>> 4c2722a3fcadace9ef261cd86d95e432a14f8376
        averages = get_consts_olson_98()[0]
        for i in range(1, ln):
            step_params[i] = torch.from_numpy(averages[sequence[i-1]+sequence[i]])

        self.geom_params = Geometrical_Parameters(local_params=step_params)
        self._set_pair_params_list()
        self.movable_steps = torch.full((len(self.pairs_list),), movable)

<<<<<<< HEAD
    def analyze_trajectory(self,trajectory):
=======
    def analyze_trajectory(self, trajectory):
>>>>>>> 4c2722a3fcadace9ef261cd86d95e432a14f8376
        '''Runs analysis of all frames in trajectory based on previously generated pairs list.'''
        try:
            # Skip tqdm for 0–1 iterable frames (common for static structures; avoids a pointless 1/1 bar).
            n_frames = len(trajectory) if hasattr(trajectory, '__len__') else None
            show_pbar = n_frames is None or n_frames > 1
            for ts in tqdm(trajectory, disable=not show_pbar):
                self.geom_params.trajectory.cur_step += 1

                for i in range(len(self.nucleotides)):
                    s_frame_residue, e_frame_residue = self.nucleotides[i]._get_frame_residues()
                    R, o = get_base_ref_frame(s_frame_residue, e_frame_residue)
                    self.nucleotides.ref_frames[i] = R
                    self.nucleotides.origins[i] = o
                for pair in self.pairs_list:
                    pair.get_pair_params()
                self.get_geom_params()
        except KeyboardInterrupt:
            pass
        step_sl = self.geom_params.trajectory.cur_step
        self.geom_params.trajectory = self.geom_params.trajectory[:step_sl]

        self.geom_params.trajectory.cur_step = 0

<<<<<<< HEAD
    def append_structures(self,structures,first_step_params=torch.from_numpy(BDNA_step[6:]),copy=True):
=======
    def append_structures(self, structures, first_step_params=torch.from_numpy(BDNA_step[6:]), copy=True):
>>>>>>> 4c2722a3fcadace9ef261cd86d95e432a14f8376
        '''Is called by CG_Structure.append_structures and combines provided DNA structures.

            Attributes:

            **structures** - list of structures to append.

            **first_step_params** - tensor of step parameters that are assigned to steps between last pair in n-th DNA structure and first pair in n+1-th DNA structure.
            Default values - BDNA step.

            **copy** - If True - copies all given structures. Default - True.'''
        if hasattr(self, 'geom_params'):
            step_params = self.step_params.clone()
        else:
            step_params = torch.empty((0, 6), dtype=torch.double)
        if copy:
            structures = [structure.copy() for structure in structures]
        ln = len(self.nucleotides)
        for structure in structures:
            structure.geom_params._auto_rebuild_sw = False
<<<<<<< HEAD
            structure.step_params[0,:] = first_step_params
            step_params = torch.cat([step_params,structure.step_params])
=======
            structure.step_params[0, :] = first_step_params
            step_params = torch.cat([step_params, structure.step_params])
>>>>>>> 4c2722a3fcadace9ef261cd86d95e432a14f8376

            for i in range(len(structure.pairs_list)):
                structure.pairs_list.lead_nucl_inds[i] += ln
                structure.pairs_list.lag_nucl_inds[i] += ln
            self.pairs_list += structure.pairs_list
            self.nucleotides += structure.nucleotides
            structure.geom_params._auto_rebuild_sw = structure.geom_params.auto_rebuild
        self.pairs_list.nucleotides_storage = self.nucleotides
        step_params[0] = torch.zeros(6)
        self.geom_params = Geometrical_Parameters(local_params=step_params)
        self._set_pair_params_list()
        self.movable_steps = torch.cat([self.movable_steps]+[structure.movable_steps for structure in structures])
        self.movable_steps[0] = False

        return self

<<<<<<< HEAD
    def transfer_trajectory_to_h5(self,filename,mode='r+',keep_single_old_frame=None,**datasets_kwards):
        old_traj = self.geom_params.trajectory
        self.geom_params.trajectory = H5_Trajectory(filename, len(self.pairs_list),mode=mode,**datasets_kwards)
        if keep_single_old_frame is not None:
            old_traj.cur_step = keep_single_old_frame
            for attr in self.geom_params.trajectory.attrs_names:
                setattr(self.geom_params.trajectory,attr,getattr(old_traj,attr))
        elif mode != 'r' and mode != 'r+':
=======
    def transfer_trajectory_to_h5(self, filename, mode='r+', keep_old_frames=True, **datasets_kwards):
        old_traj = self.geom_params.trajectory
        self.geom_params.trajectory = H5_Trajectory(filename, len(self.pairs_list), mode=mode, **datasets_kwards)
        if mode != 'r':
>>>>>>> 4c2722a3fcadace9ef261cd86d95e432a14f8376
            self.geom_params.trajectory.extend(old_traj)
        if isinstance(old_traj, H5_Trajectory):
            old_traj.file.close()

    def transfer_trajectory_to_memory(self):
        if isinstance(self.geom_params.trajectory, Tensor_Trajectory):
            raise TypeError('Trajectory is already stored in memory')
        old_traj = self.geom_params.trajectory

        self.geom_params.trajectory = Tensor_Trajectory(self.geom_params.dtype, 0,
                                                        self.origins.shape[0], mod_Tensor, self.geom_params,
                                                        attrs_names=None, shapes=None)
        self.geom_params.trajectory.extend(old_traj)
        self.geom_params.trajectory.cur_step = old_traj.cur_step
        if isinstance(old_traj, H5_Trajectory):
            old_traj.file.close()

    def move_to_coord_center(self):
        '''Transforms origins and reference frames so that the first step origin is located in the start of coordinates and reference frame is identity matrix.'''
        self.origins -= self.origins[0].clone()
        ref_R = self.ref_frames[0].clone()
<<<<<<< HEAD
        self.ref_frames = torch.matmul(ref_R.T,self.ref_frames)
        self.origins = torch.matmul(self.origins.reshape(-1,1,3),ref_R)
=======
        self.ref_frames = torch.matmul(ref_R.T, self.ref_frames)
        self.origins = torch.matmul(self.origins.reshape(-1, 1, 3), ref_R)
>>>>>>> 4c2722a3fcadace9ef261cd86d95e432a14f8376

    def get_dataframe(self):
        '''Creates pandas DataFrame that contains basic information about pairs and their geometrical parameters.

            Returns:

            pandas.DataFrame object'''
<<<<<<< HEAD
        pairs_data = [(pair.lead_nucl.resid,pair.lead_nucl.segid,pair.lead_nucl.restype,
              pair.lag_nucl.restype,pair.lag_nucl.segid,pair.lag_nucl.resid) for pair in self.pairs_list]
        labels = ['resid1','segid1','restype1','restype2','segid2','resid2']
        df = pd.DataFrame(pairs_data,columns = labels)
        labels = ['Shear','Stretch','Stagger','Buckle','Prop-Tw','Opening','Shift','Slide','Rise','Tilt','Roll','Twist']
        params = np.hstack([self.pairs_params,self.steps_params])
        params_df = pd.DataFrame(np.hstack([self.pairs_params,self.steps_params]),columns=labels)
        return pd.concat([df,params_df],axis=1)

    def to(self,device):
=======
        pairs_data = [(pair.lead_nucl.resid, pair.lead_nucl.segid, pair.lead_nucl.restype,
                       pair.lag_nucl.restype, pair.lag_nucl.segid, pair.lag_nucl.resid) for pair in self.pairs_list]
        labels = ['resid1', 'segid1', 'restype1', 'restype2', 'segid2', 'resid2']
        df = pd.DataFrame(pairs_data, columns=labels)
        labels = ['Shear', 'Stretch', 'Stagger', 'Buckle', 'Prop-Tw', 'Opening', 'Shift', 'Slide', 'Rise', 'Tilt', 'Roll', 'Twist']
        params = np.hstack([self.pairs_params, self.steps_params])
        params_df = pd.DataFrame(np.hstack([self.pairs_params, self.steps_params]), columns=labels)
        return pd.concat([df, params_df], axis=1)

    def to(self, device):
>>>>>>> 4c2722a3fcadace9ef261cd86d95e432a14f8376
        '''Moves tensors of supplemental parameters of DNA to a given device.'''
        self.radii = self.radii.to(device)
        self.eps = self.eps.to(device)
        self.charges = self.charges.to(device)

    def copy(self):
        '''Creates a deep copy of self.'''
        new = DNA_Structure()
        new.radii = self.radii.clone()
        new.eps = self.eps.clone()
        new.charges = self.charges.clone()
        new.movable_steps = self.movable_steps.clone()
        new.geom_params = self.geom_params.copy()
        new.pairs_list = self.pairs_list.copy()
        new.nucleotides = self.nucleotides.copy()
<<<<<<< HEAD

        return new

    def parse_pairs(self,pairs_in_structure):
        '''Is called in analyze_dna method if list of pairs is provided. Creates a pair list based on given information.'''
        self.pairs_list = Pairs_Storage(Base_Pair,self.nucleotides)
        for pair_data in pairs_in_structure:
            resid1,segid1,resid2,segid2 = pair_data

            nucl1 = self.nucleotides[[s==segid1 and r==resid1 for s,r in zip(self.nucleotides.segids,self.nucleotides.resids)]]

            nucl2 = self.nucleotides[[s==segid2 and r==resid2 for s,r in zip(self.nucleotides.segids,self.nucleotides.resids)]]
=======

        return new

    def parse_pairs(self, pairs_in_structure):
        '''Is called in analyze_dna method if list of pairs is provided. Creates a pair list based on given information.'''
        self.pairs_list = Pairs_Storage(Base_Pair, self.nucleotides)
        for pair_data in pairs_in_structure:
            resid1, segid1, resid2, segid2 = pair_data
            nucl1 = self.nucleotides[[s == segid1 and r == resid1 for s, r in zip(self.nucleotides.segids, self.nucleotides.resids)]]
            if len(nucl1) != 1:
                raise ValueError(f'Expected exactly one nucleotide for ({resid1}, {segid1}), got {len(nucl1)}')
            nucl1 = nucl1[0]

            nucl2 = self.nucleotides[[s == segid2 and r == resid2 for s, r in zip(self.nucleotides.segids, self.nucleotides.resids)]]
            if len(nucl2) != 1:
                raise ValueError(f'Expected exactly one nucleotide for ({resid2}, {segid2}), got {len(nucl2)}')
            nucl2 = nucl2[0]
>>>>>>> 4c2722a3fcadace9ef261cd86d95e432a14f8376

            new_base_pair = Base_Pair(self.pairs_list,
                                      lead_nucl=nucl1, lag_nucl=nucl2)
            new_base_pair.get_pair_params()

<<<<<<< HEAD
    def save_to_h5(self,file,**dataset_kwards):
        group = file.create_group('DNA_data')
        self.nucleotides.save_to_h5(group,'nucleotides',**dataset_kwards)
        self.pairs_list.save_to_h5(group,'pairs',**dataset_kwards)
        group.create_dataset('step_params',data=self.step_params,**dataset_kwards)
        group.create_dataset('origins',data=self.origins,**dataset_kwards)
        group.create_dataset('ref_frames',data=self.ref_frames,**dataset_kwards)
        group.create_dataset('movable_steps',data=self.movable_steps,dtype=bool)

    def load_from_h5(self,file):
        data = file['DNA_data']

        self.nucleotides = Nucleotides_Storage(Nucleotide,self.u)
        self.pairs_list = Pairs_Storage(Base_Pair,self.nucleotides)

        self.pairs_list.load_from_h5(data,'pairs')
        self.nucleotides.load_from_h5(data,'nucleotides')
        self.geom_params = Geometrical_Parameters(local_params=torch.tensor(data['step_params']),
                                                 origins = torch.tensor(data['origins']),
                                                 ref_frames = torch.tensor(data['ref_frames'])
                                                 )
        self.movable_steps = torch.tensor(data['movable_steps'])
=======
    def save_to_h5(self, file, **dataset_kwards):

        group = file.create_group('DNA_data')
        self.nucleotides.save_to_h5(group, 'nucleotides', **dataset_kwards)
        self.pairs_list.save_to_h5(group, 'pairs', **dataset_kwards)
        group.create_dataset('step_params', data=self.step_params, **dataset_kwards)
        group.create_dataset('origins', data=self.origins, **dataset_kwards)
        group.create_dataset('ref_frames', data=self.ref_frames, **dataset_kwards)
        group.create_dataset('movable_steps', data=self.movable_steps, dtype=bool)

    def load_from_h5(self, file):
        data = file['DNA_data']

        self.nucleotides = Nucleotides_Storage(Nucleotide, self.u)
        self.pairs_list = Pairs_Storage(Base_Pair, self.nucleotides)

        self.pairs_list.load_from_h5(data, 'pairs')
        self.nucleotides.load_from_h5(data, 'nucleotides')
        self.geom_params = Geometrical_Parameters(
            local_params=torch.tensor(np.asarray(data['step_params'])),
            origins=torch.tensor(np.asarray(data['origins'])),
            ref_frames=torch.tensor(np.asarray(data['ref_frames'])),
        )
        if 'movable_steps' in data:
            self.movable_steps = torch.tensor(np.asarray(data['movable_steps']))
        else:
            # Older cg_*.h5 files predate this dataset; default matches build_from_u(movable=False).
            self.movable_steps = torch.zeros(len(self.pairs_list), dtype=torch.bool)
>>>>>>> 4c2722a3fcadace9ef261cd86d95e432a14f8376
        self._set_pair_params_list()

    def _set_pair_params_list(self):
        '''Fetches parameters from individual Pair objects'''
        self.radii = torch.tensor(self.pairs_list.radii)
        self.eps = torch.tensor([self.pairs_list.epsilons])
        self.charges = torch.tensor([self.pairs_list.charges])

<<<<<<< HEAD
    def _getter(self,attr):
        return getattr(self.geom_params,attr)

    def _setter(self,value,attr):
        setattr(self.geom_params,attr,value)

    def _set_property(attr):
        return property(lambda self: self._getter(attr=attr),lambda self,value: self._setter(value,attr=attr))
=======
    def _getter(self, attr):
        return getattr(self.geom_params, attr)

    def _setter(self, value, attr):
        setattr(self.geom_params, attr, value)

    def _set_property(attr):
        return property(lambda self: self._getter(attr=attr), lambda self, value: self._setter(value, attr=attr))
>>>>>>> 4c2722a3fcadace9ef261cd86d95e432a14f8376

    @property
    def trajectory(self):
        return self.geom_params.trajectory

    step_params = _set_property('local_params')
    ref_frames = _set_property('ref_frames')

    @property
    def origins(self):
        return self.geom_params.origins

    @origins.setter
    def origins(self, value):
        self.geom_params.origins = value

    def __repr__(self):
        return f'<DNA structure with {len(self.pairs_list)} nucleotide pairs>'

<<<<<<< HEAD
    def __getitem__(self,sl):
        ''''''
        attrs = self.__dict__.copy()
        for attr in ('pairs_list', 'geom_params'):
            attrs[attr] = getattr(self,attr)[sl]
=======
    def __getitem__(self, sl):
        ''''''
        attrs = self.__dict__.copy()
        for attr in ('pairs_list', 'geom_params'):
            attrs[attr] = getattr(self, attr)[sl]
>>>>>>> 4c2722a3fcadace9ef261cd86d95e432a14f8376

        for attr in ('radii', 'eps', 'charges'):
            if sl.step < 0:
                it = getattr(self, attr).flip(dims=(0,))
                new_sl = slice(sl.start, sl.stop, -1*sl.step)
                attrs[attr] = it[new_sl]
            else:
<<<<<<< HEAD
                attrs[attr] = getattr(self,attr)[sl]

        return DNA_Structure(**attrs)
=======
                attrs[attr] = getattr(self, attr)[sl]

        return DNA_Structure(**attrs)
>>>>>>> 4c2722a3fcadace9ef261cd86d95e432a14f8376
