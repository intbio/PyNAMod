import pandas as pd
import numpy as np
import torch
import nglview as nv
from Bio.Seq import Seq

from tqdm.auto import tqdm
from pynamod.energy.energy_constants import get_consts_olson_98,BDNA_step
from pynamod.geometry.geometrical_parameters import Geometrical_Parameters
from pynamod.atomic_analysis.nucleotides_parser import get_all_nucleotides, Nucleotide,get_base_ref_frame
from pynamod.atomic_analysis.pairs_parser import get_pairs, Base_Pair
from pynamod.atomic_analysis.structures_storage import Nucleotides_Storage,Pairs_Storage

class DNA_Structure:
    '''Class that stores information about DNA, pairs in it. Geometrical parameters in this class are links to the other class (All_Coords) that manages them.
    
        Main attributes of this class are:
        
        - **pairs_list** - list of Pair class objects. 
        - **origins**, **ref_frames**, **step_params** - properties that return tensors from All_Coords class with parameters for this DNA structure.'''
    def __init__(self,**kwards):
        self.pairs_list = []
        self.traj_step = 1
        for name,value in kwards.items():
            setattr(self,name,value)
                
    def build_from_u(self,leading_strands,pairs_in_structure = None,traj_len=1,sel=None,overwrite_existing_dna=False):
        '''This method is called by CG_Structure.analyze_dna and runs full analysis of atomic structure.'''
        self.nucleotides = get_all_nucleotides(self,leading_strands,sel)

        if pairs_in_structure is not None:
            self.pairs_list = self.parse_pairs(pairs_in_structure)
        else:
            self.pairs_list = get_pairs(self)
        if not self.pairs_list:
            raise TypeError('No pairs were found')
            
        self.get_geom_params(traj_len,overwrite_existing_dna)
        self._set_pair_params_list()
        
    def get_geom_params(self,traj_len=1,overwrite_existing_dna=False):
        '''This method prepairs reference frames and orgins frim pairs, than creates object of All_Coords or updates existing one.'''
        ref_frames = torch.stack([pair.Rm for pair in self.pairs_list])
        
        origins = torch.vstack([pair.om for pair in self.pairs_list]).reshape(-1,1,3)
        if hasattr(self,'geom_params') and not overwrite_existing_dna:
            self.geom_params.get_new_params_set(ref_frames=ref_frames,origins=origins)
        else:
            self.geom_params = Geometrical_Parameters(ref_frames=ref_frames,origins=origins,traj_len = traj_len)
    
    
    def generate(self,sequence,radius=10,charge=-2,eps=0.5):
        '''This method is called by CG_Structure.build_dna and creates linear DNA based on given sequence and BDNA step.'''
        sequence = sequence.upper()
        rev_sequence = Seq(sequence).reverse_complement()
        ln = len(sequence)
        
        self.nucleotides = Nucleotides_Storage(Nucleotide,None)
        self.nucleotides.restypes = list(sequence+rev_sequence)
        self.nucleotides.resids = list(range(ln))*2
        self.nucleotides.segids = ['A']*ln + ['B']*ln
        self.nucleotides.leading_strands = [True]*ln + [False]*ln
        self.nucleotides.ref_frames = torch.zeros(ln,3,3,dtype=torch.double)
        self.nucleotides.origins = torch.zeros(ln,1,3,dtype=torch.double)
        self.nucleotides.s_residues = [None]*(ln*2)
        self.nucleotides.e_residues = [None]*(ln*2)
        self.nucleotides.base_pairs = [None]*(ln*2)
        
        self.pairs_list = Pairs_Storage(Base_Pair,self.nucleotides)
        self.pairs_list.lead_nucl_inds = list(range(ln))
        self.pairs_list.lag_nucl_inds = [i+ln for i in range(ln)]
        self.pairs_list.radii = [radius]*ln
        self.pairs_list.charges = [charge]*ln
        self.pairs_list.epsilons = [eps]*ln
        pair_params = torch.zeros(2,6,dtype=torch.double)
        pair_params[1] = torch.from_numpy(BDNA_step[:6])
        self.pairs_list.geom_paramss = [Geometrical_Parameters(local_params=pair_params.clone()) for i in range(ln)]
        
        step_params = torch.zeros(ln,6,dtype=torch.double)
        averages = get_consts_olson_98()[0]
        for i in range(1,ln):
            step_params[i] = torch.from_numpy(averages[sequence[i-1]+sequence[i]])

        self.geom_params = Geometrical_Parameters(local_params=step_params)
        self._set_pair_params_list()
    
    def analyze_trajectory(self,trajectory):
        '''Runs analysis of all frames in trajectory based on previously generated pairs list.'''
        try:
            for ts in tqdm(trajectory):
                self.geom_params.trajectory.cur_step += 1

                for i in range(len(self.nucleotides)):
                    R,o = get_base_ref_frame(self.nucleotides.s_residues[i],self.nucleotides.e_residues[i])
                    self.nucleotides.ref_frames[i] = R
                    self.nucleotides.origins[i] = o
                for pair in self.pairs_list:
                     pair.get_pair_params()
                self.get_geom_params()
        except KeyboardInterrupt:
            pass
        step_sl = self.geom_params.trajectory.cur_step
        self.geom_params.trajectory.origins_traj = self.geom_params.trajectory.origins_traj[:step_sl]
        self.geom_params.trajectory.ref_frames_traj = self.geom_params.trajectory.ref_frames_traj[:step_sl]
        self.geom_params.trajectory.local_params_traj = self.geom_params.trajectory.local_params_traj[:step_sl]
        
        self.geom_params.trajectory.cur_step = 0
        
        
    
    def append_structures(self,structures,first_step_params=torch.from_numpy(BDNA_step[6:]),copy=True):
        '''Is called by CG_Structure.append_structures and combines provided DNA structures.
            
            Attributes:
            
            **structures** - list of structures to append.
            
            **first_step_params** - tensor of step parameters that are assigned to steps between last pair in n-th DNA structure and first pair in n+1-th DNA structure.
            Default values - BDNA step.
            
            **copy** - If True - copies all given structures. Default - True.'''
        if hasattr(self,'geom_params'):
            step_params = self.step_params.clone()
        else:
            step_params = torch.empty((0,6),dtype=torch.double)
        if copy:
            structures = [structure.copy() for structure in structures]
        ln = len(self.nucleotides)
        for structure in structures:
            structure.geom_params._auto_rebuild_sw = False
            structure.step_params[0,:] = first_step_params
            step_params = torch.cat([step_params,structure.step_params])
        
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
        
        return self
        
        
    def move_to_coord_center(self):
        '''Transforms origins and reference frames so that the first step origin is located in the start of coordinates and reference frame is identity matrix.'''
        self.origins -= self.origins[0].clone()
        ref_R = self.ref_frames[0].clone()
        self.ref_frames = torch.matmul(ref_R.T,self.ref_frames)
        self.origins = torch.matmul(self.origins.reshape(-1,1,3),ref_R)
        
        
    def get_dataframe(self):
        '''Creates pandas DataFrame that contains basic information about pairs and their geometrical parameters.
        
            Returns:
            
            pandas.DataFrame object'''
        pairs_data = [(pair.lead_nucl.resid,pair.lead_nucl.segid,pair.lead_nucl.restype,
              pair.lag_nucl.restype,pair.lag_nucl.segid,pair.lag_nucl.resid) for pair in self.pairs_list]
        labels = ['resid1','segid1','restype1','restype2','segid2','resid2']
        df = pd.DataFrame(pairs_data,columns = labels)
        labels = ['Shear','Stretch','Stagger','Buckle','Prop-Tw','Opening','Shift','Slide','Rise','Tilt','Roll','Twist']
        params = np.hstack([self.pairs_params,self.steps_params])
        params_df = pd.DataFrame(np.hstack([self.pairs_params,self.steps_params]),columns=labels)
        return pd.concat([df,params_df],axis=1)
        
        
    def to(self,device):
        '''Moves tensors of supplemental parameters of DNA to a given device.'''
        self.radii = self.radii.to(device)
        self.eps = self.eps.to(device)
        self.charges = self.charges.to(device)
        
        
    def copy(self):
        '''Creates a deep copy of self.'''
        new = DNA_Structure()
        new.cg_structure = self.cg_structure
        new.radii = self.radii.clone()
        new.eps = self.eps.clone()
        new.charges = self.charges.clone()
        new.geom_params = self.geom_params.copy()
        new.pairs_list = self.pairs_list.copy()
        new.nucleotides = self.nucleotides.copy()
        
        return new
    
    def parse_pairs(self,pairs_in_structure):
        '''Is called in analyze_dna method if list of pairs is provided. Creates a pair list based on given information.'''
        self.pairs_list = []
        for pair_data in pairs_in_structure:
            resid1,segid1,resid2,segid2 = pair_data
            nucl1 = self.nucleotides[self.nucleotides.segids==segid1]

    def save_to_h5(self,file,**dataset_kwards):
        
        group = file.create_group('DNA_data')
        self.nucleotides.save_to_h5(group,'nucleotides',**dataset_kwards)
        self.pairs_list.save_to_h5(group,'pairs',**dataset_kwards)
        group.create_dataset('step_params',data=self.step_params,**dataset_kwards)
        group.create_dataset('origins',data=self.origins,**dataset_kwards)
        group.create_dataset('ref_frames',data=self.ref_frames,**dataset_kwards)
        
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
        self._set_pair_params_list()
            
    def _set_pair_params_list(self):
        '''Fetches parameters from individual Pair objects'''
        self.radii = torch.tensor(self.pairs_list.radii)
        self.eps = torch.tensor([self.pairs_list.epsilons])
        self.charges = torch.tensor([self.pairs_list.charges])

    def __getter(self,attr):
        return getattr(self.geom_params,attr)
    
    def __setter(self,value,attr):
        setattr(self.geom_params,attr,value)
        
    def __set_property(attr):
        return property(lambda self: self.__getter(attr=attr),lambda self,value: self.__setter(value,attr=attr))
        
    def traj(self):
        for step in range(0,self.geom_params.trajectory.get_len(),self.traj_step):
            self.geom_params.trajectory.cur_step = step
            yield step
    
    @property
    def trajectory(self):
        self.geom_params.trajectory.cur_step = 0
        return self.traj()
    
    
    step_params = __set_property('local_params')
    ref_frames = __set_property('ref_frames')

    @property
    def origins(self):
        return self.geom_params.origins
    
    @origins.setter
    def origins(self,value):
        self.geom_params.origins = value
    
    def __repr__(self):
        return f'<DNA structure with {len(self.pairs_list)} nucleotide pairs>'
    
    
    def __getitem__(self,sl):
        ''''''
        attrs = self.__dict__.copy()
        for attr in ('pairs_list', 'geom_params'):
            attrs[attr] = getattr(self,attr)[sl]
        
        for attr in ( 'radii', 'eps', 'charges'):
            if sl.step < 0:
                it = getattr(self,attr).flip(dims=(0,))
                new_sl = slice(sl.start,sl.stop,-1*sl.step)
                attrs[attr] = it[new_sl]
            else:
                attrs[attr] = getattr(self,attr)[sl]
    
        return DNA_Structure(**attrs)
        
    
        