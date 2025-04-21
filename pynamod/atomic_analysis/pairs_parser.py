import numpy as np
import torch
import pickle
from itertools import combinations
from more_itertools import pairwise
from scipy.spatial.transform import Rotation as R
from pynamod.geometry.geometrical_parameters import Geometrical_Parameters
from pynamod.atomic_analysis.structures_storage import Pairs_Storage

from importlib.resources import files

path = files('pynamod').joinpath('atomic_analysis/classifier.pkl')
with open(path, 'rb') as f:
    classifier = pickle.load(f)


class Base_Pair:
    def __init__(self, storage_class,lead_nucl_ind=None,
                 lag_nucl_ind=None,ind=None,lead_nucl=None,lag_nucl=None,radius=10,charge=-2,eps=0.5,geom_params=None):
        if lead_nucl:
            lead_nucl_ind = lead_nucl.ind
            lag_nucl_ind = lag_nucl.ind

        if ind is None:
            ind = len(storage_class)
            storage_class.append(lead_nucl_ind,lag_nucl_ind,radius,charge,eps,geom_params)
            if storage_class.nucleotides_storage[lag_nucl_ind].leading_strand:
                self.lead_nucl_ind, self.lag_nucl_ind = self.lag_nucl_ind, self.lead_nucl_ind
        else:
            lead_nucl_ind = storage_class.lead_nucl_inds[ind]
            lag_nucl_ind = storage_class.lag_nucl_inds[ind]


        self.storage_class = storage_class
        self.ind = ind
        #self.pair_name = ''.join(sorted((lead_nucl.restype, lag_nucl.restype))).upper()
        

        
    def __lt__(self, other):
        return self.lead_nucl.__lt__(other.lead_nucl)
    
    def __repr__(self):
        return f'<Nucleotides pair with resids {self.lead_nucl.resid}, {self.lag_nucl.resid}, and segids {self.lead_nucl.segid}, {self.lag_nucl.segid}>'
    
    def update_references(self):
        self.lead_nucl.base_pair = self.lag_nucl.base_pair = self
    
    def get_pair_params(self):
        ori = torch.vstack([self.lead_nucl.origin,self.lag_nucl.origin])
        r_frames = torch.stack([self.lead_nucl.ref_frame,self.lag_nucl.ref_frame])

        if self.geom_params:
            self.geom_params.get_new_params_set(ref_frames=r_frames,origins=ori)
        else:
            self.geom_params = Geometrical_Parameters(ref_frames = r_frames, origins = ori,
                                                  pair_params=True)            

    
    def copy(self,**kwards):
        new = Base_Pair(lead_nucl = self.lead_nucl.copy(),lag_nucl = self.lag_nucl.copy(),radius=self.radius,charge=self.charge,eps=self.eps,dna_structure=self.dna_structure,geom_params=self.geom_params)
        new.update_references()
        for name,value in kwards.items():
            setattr(new,name,value)
        return new
    

    def get_params(self,attr):
        return getattr(self.geom_params,attr)[0]
    
    def set_params(self,value,attr):
        getattr(self.geom_params,attr)[0] = value
        
    def __setter(self,value,attr):
        getattr(self.storage_class,self.storage_class.get_name(attr))[self.ind] = value
        
    def __getter(self,attr):
        return getattr(self.storage_class,self.storage_class.get_name(attr))[self.ind]
        
    def __set_property(attr):
        setter = lambda self,value: self.__setter(value,attr=attr)
        getter = lambda self: self.__getter(attr=attr)
        return property(fset=setter,fget=getter)

    lead_nucl_ind = __set_property('lead_nucl_ind')
    lag_nucl_ind = __set_property('lag_nucl_ind')
    radius = __set_property('radius')
    charge = __set_property('charge')
    epsilon = __set_property('epsilon')
    geom_params = __set_property('geom_params')
    
        
    pair_params = property(fget=lambda self: self.get_params(attr='local_params'),fset=lambda self,value: self.set_params(attr='local_params'))
    Rm = property(fget=lambda self: self.get_params(attr='ref_frames'),fset=lambda self,value: self.set_params(attr='ref_frames'))
    om = property(fget=lambda self: self.get_params(attr='origins'),fset=lambda self,value: self.set_params(attr='origins'))
    
    @property
    def pair_name(self):
        if isinstance(self.storage_class.nucleotides_storage[int(self.lead_nucl_ind)].restype,bytes):
            return ''.join(sorted((str(self.storage_class.nucleotides_storage[int(self.lead_nucl_ind)].restype,"utf-8"), 
                               str(self.storage_class.nucleotides_storage[int(self.lag_nucl_ind)].restype,"utf-8")))).upper()
        else:
            return ''.join(sorted((str(self.storage_class.nucleotides_storage[int(self.lead_nucl_ind)].restype), 
                               str(self.storage_class.nucleotides_storage[int(self.lag_nucl_ind)].restype)))).upper()
    
    @property
    def lead_nucl(self):
        return self.storage_class.nucleotides_storage[self.lead_nucl_ind]
    
    @property
    def lag_nucl(self):
        return self.storage_class.nucleotides_storage[self.lag_nucl_ind]

def check_if_pair(lead_nucl,lag_nucl):
    if lag_nucl.leading_strand:
        lead_nucl, lag_nucl = lag_nucl, lead_nucl
    if ''.join(sorted((lead_nucl.restype, lag_nucl.restype))).upper() in ('AT', 'CG', 'AU'):
        dist = np.linalg.norm(lead_nucl.origin - lag_nucl.origin)
        if dist < 2:        
            pred_params2 = R.align_vectors(lead_nucl.ref_frame,lag_nucl.ref_frame)[0].as_euler('zyx', degrees=True)
            pred_params2[2] += (pred_params2[2] < 0) * 360
            pred_params2 = np.append(pred_params2, dist)

            if bool(classifier.predict(pred_params2.reshape(1,-1))):
                return True
    return False
    
def get_pairs(dna_structure):
    '''
    Get pairs of nucleotides for multiple structures based on RandomForest classifier algoritm
    '''
    nucleotides = dna_structure.nucleotides
    pairs = Pairs_Storage(Base_Pair,nucleotides)
    for pair_candidate in combinations(nucleotides, 2):
        if check_if_pair(*pair_candidate):
            base_pair = Base_Pair(pairs,lead_nucl=pair_candidate[0],lag_nucl=pair_candidate[1])
            base_pair.get_pair_params()
            base_pair.update_references()
    
    return fix_missing_pairs(dna_structure,pairs)

    
def fix_missing_pairs(dna_structure,pairs):
    '''
Add missing pairs by context after classifier algorithm.
Fix the cases
A - T
G   C
T - A
by assuming the middle nucleotides are a pair.
-----
    '''
    nucleotides = dna_structure.nucleotides
    for nucleotide in nucleotides[:-1]:
        if not nucleotide.base_pair and nucleotide.leading_strand and nucleotide.next_nucleotide and nucleotide.previous_nucleotide:
            if nucleotide.next_nucleotide.base_pair and nucleotide.previous_nucleotide.base_pair:
                pos_pair_nucleotide_1 = nucleotide.next_nucleotide.base_pair.lag_nucl.next_nucleotide
                pos_pair_nucleotide_2 = nucleotide.previous_nucleotide.base_pair.lag_nucl.previous_nucleotide

                if pos_pair_nucleotide_1.ind == pos_pair_nucleotide_2.ind:

                    if ''.join(sorted((nucleotide.restype, pos_pair_nucleotide_1.restype))).upper() in ('AT', 'CG', 'AU'):
                        new_base_pair = Base_Pair(pairs,lead_nucl=nucleotide, lag_nucl=pos_pair_nucleotide_1)
                        new_base_pair.get_pair_params()
                        new_base_pair.update_references()
    
    return pairs.sort()