import pandas as pd
import numpy as np
import torch
import nglview as nv
from Bio.Seq import Seq

from tqdm.auto import tqdm
from pynamod.energy.energy_constants import get_consts_olson_98,BDNA_step
from pynamod.geometry.geometrical_parameters import Geometrical_Parameters
from pynamod.geometry.all_coords import All_Coords
from pynamod.atomic_analysis.nucleotides_parser import get_all_nucleotides, Nucleotide
from pynamod.atomic_analysis.pairs_parser import get_pairs, Base_Pair

class DNA_Structure:
    '''Class that stores information about DNA, pairs in it. Geometrical parameters in this class are links to the other class (All_Coords) that manages them.
    
        Main attributes of this class are:
        
        - **pairs_list** - list of Pair class objects. 
        - **origins**, **ref_frames**, **step_params** - properties that return tensors from All_Coords class with parameters for this DNA structure.'''
    def __init__(self,**kwards):
        self.pairs_list = []
        for name,value in kwards.items():
            setattr(self,name,value)
                
    def build_from_u(self,leading_strands,pairs_in_structure = None,sel=None):
        '''This method is called by CG_Structure.analyze_dna and runs full analysis of atomic structure.'''
        self.nucleotides = get_all_nucleotides(self,leading_strands,sel)

        if pairs_in_structure is not None:
            self.pairs_list = self.parse_pairs(pairs_in_structure)
        else:
            self.pairs_list = get_pairs(self)
            
        if not self.pairs_list:
            raise TypeError('No pairs were found')
            
        self.get_geom_params()
        self._set_pair_params_list()
        
    def get_geom_params(self):
        '''This method prepairs reference frames and orgins frim pairs, than creates object of All_Coords or updates existing one.'''
        ref_frames = torch.stack([pair.Rm for pair in self.pairs_list])
        
        origins = torch.vstack([pair.om for pair in self.pairs_list]).reshape(-1,1,3)
        if self.geom_params:
            self.geom_params.get_new_params_set(ref_frames=ref_frames,origins=origins)
        else:
            self.geom_params = All_Coords(ref_frames=ref_frames,origins=origins)
    
    
    def generate(self,sequence):
        '''This method is called by CG_Structure.build_dna and creates linear DNA based on given sequence and BDNA step.'''
        self.pairs_list = []
        DNA_length = len(sequence)
        step_params = torch.zeros(DNA_length,6,dtype=torch.double)
        averages = get_consts_olson_98()[0]
        rev_sequence = Seq(sequence).reverse_complement()
        prev_lead_nucl = prev_lag_nucl = prev_pair =  None
        for i,(lead_res,lag_res) in enumerate(zip(sequence.upper(),rev_sequence.upper())):
            
            lead_nucl = Nucleotide(lead_res, i, 'A', True)
            lag_nucl = Nucleotide(lag_res, DNA_length-i, 'B', False)
            pair_params = torch.zeros(2,6,dtype=torch.double)
            pair_params[1] = torch.from_numpy(BDNA_step[:6])
            pair = Base_Pair(lead_nucl,lag_nucl,geom_params = Geometrical_Parameters(local_params=pair_params),dna_structure=self)
            lead_nucl.R,lead_nucl.o = pair.geom_params.ref_frames[0],pair.geom_params.origins[0]
            lag_nucl.R,lag_nucl.o = pair.geom_params.ref_frames[1],pair.geom_params.origins[1]
            lead_nucl.base_pair = lag_nucl.base_pair = pair
            if prev_lead_nucl:
                step_params[i] = torch.from_numpy(averages[prev_lead_nucl.restype+lead_res])
                lead_nucl.previous_nucleotide = prev_lead_nucl
                lag_nucl.previous_nucleotide = prev_lag_nucl
                prev_lead_nucl.next_nucleotide = lead_nucl
                prev_lag_nucl.next_nucleotide = lag_nucl
                
            
            self.pairs_list.append(pair)
            prev_pair = pair
            prev_lead_nucl = lead_nucl
            prev_lag_nucl = lag_nucl
        
        self.geom_params = All_Coords(local_params=step_params)
        self._set_pair_params_list()
    
    def analyze_trajectory(self,u):
        '''Runs analysis of all frames in trajectory based on previously generated pairs list.'''
        self.geom_params.test_sw = True
        for ts in tqdm(u.trajectory):
            self.geom_params._cur_tr_step += 1
            for n in self.nucleotides:
                n.get_base_ref_frame()
            for pair in self.pairs_list:
                pair.get_pair_params()
            self.get_geom_params()
        self.geom_params._cur_tr_step = 0
    
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
       
        for structure in structures:
            structure.geom_params._auto_rebuild_sw = False
            structure.step_params[0,:] = first_step_params
            step_params = torch.cat([step_params,structure.step_params])
            for pair in structure.pairs_list:
                pair.dna_structure = self
            self.pairs_list += structure.pairs_list
            structure.geom_params._auto_rebuild_sw = True
            
        step_params[0] = torch.zeros(6)
        self.geom_params = All_Coords(local_params=step_params)
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
        new.pairs_list = [pair.copy(dna_structure = new) for pair in self.pairs_list]
        
        return new
    
    def parse_pairs(self,pairs_in_structure):
        '''Is called in analyze_dna method if list of pairs is provided. Creates a pair list based on given information.'''
        pairs_list = []
        for pair_data in pairs_in_structure:
            resid1,segid1,resid2,segid2 = pair_data
            nucl1 = nucl2 = None
            for nucl in self.nucleotides:
                if not nucl1 and nucl.resid == resid1 and nucl.segid == segid1:
                    nucl1 = nucl
                elif not nucl2 and nucl.resid == resid2 and nucl.segid == segid2:
                    nucl2 = nucl
            pair = Base_pair(nucl1,nucl2,self)
            #maybe needs update
            pair.update_references()
            
    def _set_pair_params_list(self):
        '''Fetches parameters from individual Pair objects'''
        self.radii = torch.Tensor([pair.radius for pair in self.pairs_list])
        self.eps = torch.Tensor([pair.eps for pair in self.pairs_list])
        self.charges = torch.Tensor([pair.charge for pair in self.pairs_list])

    def __getter(self,attr):
        return getattr(self.geom_params,attr)
    
    def __setter(self,value,attr):
        setattr(self.geom_params,attr,value)
        
    def __set_property(attr):
        return property(lambda self: self.__getter(attr=attr),lambda self,value: self.__setter(value,attr=attr))
        
    def traj(self):
        for step in range(self.geom_params.traj_len()):
            self.geom_params._cur_tr_step = step
            yield step
    
    @property
    def trajectory(self):
        self.geom_params._cur_tr_step = 0
        return self.traj()
    
    @property
    def geom_params(self):
        return self.cg_structure.all_coords
    
    @geom_params.setter
    def geom_params(self,value):
        self.cg_structure.all_coords = value
    
    step_params = __set_property('local_params')
    ref_frames = __set_property('ref_frames')

    @property
    def origins(self):
        return self.cg_structure.all_coords.origins[:self.geom_params.len]
    
    @origins.setter
    def origins(self,value):
        self.cg_structure.all_coords.origins[:self.geom_params.len] = value
    
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
        
    
        