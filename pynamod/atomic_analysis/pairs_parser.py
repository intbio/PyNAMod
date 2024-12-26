import numpy as np
import torch
import pickle
from itertools import combinations
from more_itertools import pairwise
from scipy.spatial.transform import Rotation as R
from pynamod.geometry.bp_step_geometry import Geometrical_Parameters


with open('pynamod/atomic_analysis/classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)


class Base_Pair:
    def __init__(self, lead_nucl, lag_nucl,radius=2,charge=-2,eps=0.5,geom_params=None,dna_structure= None):
        if lag_nucl.in_leading_strand:
            lead_nucl, lag_nucl = lag_nucl, lead_nucl

        self.lead_nucl = lead_nucl
        self.lag_nucl = lag_nucl
        self.radius = radius
        self.charge = charge
        self.eps = eps
        self.dna_structure = dna_structure
        self.geom_params = geom_params

        self.pair_name = ''.join(sorted((lead_nucl.restype, lag_nucl.restype))).upper()
        

        
    def __lt__(self, other):
        return self.lead_nucl.__lt__(other.lead_nucl)
    
    def __repr__(self):
        return f'<Nucleotides pair with resids {self.lead_nucl.resid}, {self.lag_nucl.resid}, and segids {self.lead_nucl.segid}, {self.lag_nucl.segid}>'
    
    def update_references(self):
        self.lead_nucl.base_pair = self.lag_nucl.base_pair = self
    
    def check_if_pair(self):
        if self.pair_name in ('AT', 'CG', 'AU'):
            dist = np.linalg.norm(self.lead_nucl.o - self.lag_nucl.o)
            if dist < 2:
                pred_params = R.align_vectors(self.lead_nucl.R, self.lag_nucl.R)[0].as_euler('zyx', degrees=True)
                pred_params[2] += (pred_params[2] < 0) * 360
                pred_params = np.append(pred_params, dist)
                if bool(classifier.predict(pred_params.reshape(1,-1))):

                    self.update_references()
                    return True
        return False
    
    def get_pair_params(self):
        ori = torch.vstack([self.lead_nucl.o,self.lag_nucl.o])
        r_frames = torch.stack([self.lead_nucl.R,self.lag_nucl.R])

        if self.geom_params:
            self.geom_params.get_new_params_set(ref_frames=r_frames,origins=ori)
        else:
            self.geom_params = Geometrical_Parameters(ref_frames = r_frames, origins = ori,
                                                  pair_params=True,cur_step_cls='dna_structure.geom_params',traj_len=self.dna_structure.traj_len)            


    def get_index(self,dna_structure=None):
        if not dna_structure:
            dna_structure = self.dna_structure
        return dna_structure.pairs_list.index(self) 
    
    def copy(self,**kwards):
        new = Base_Pair(lead_nucl = self.lead_nucl.copy(),lag_nucl = self.lag_nucl.copy(),radius=self.radius,charge=self.charge,eps=self.eps,dna_structure=self.dna_structure)
        new.update_references()
        for name,value in kwards.items():
            setattr(new,name,value)
        return new
    

    def get_params(self,attr):
        return getattr(self.geom_params,attr)[0]
    
    def set_params(self,value,attr):
        getattr(self.geom_params,attr)[0] = value
        
    pair_params = property(fget=lambda self: self.get_params(attr='local_params'),fset=lambda self,value: self.set_params(attr='local_params'))
    Rm = property(fget=lambda self: self.get_params(attr='ref_frames'),fset=lambda self,value: self.set_params(attr='ref_frames'))
    om = property(fget=lambda self: self.get_params(attr='origins'),fset=lambda self,value: self.set_params(attr='origins'))
    
    
def get_pairs(dna_structure):
    '''
    Get pairs of nucleotides for multiple structures based on RandomForest classifier algoritm
    '''
    nucleotides = dna_structure.nucleotides
    pairs_list = []
    for pair_candidate in combinations(nucleotides, 2):
        
        base_pair = Base_Pair(*pair_candidate,dna_structure=dna_structure)
        if base_pair.check_if_pair():
            base_pair.get_pair_params()
            pairs_list.append(base_pair)
    
    return fix_missing_pairs(dna_structure,pairs_list)

    
def fix_missing_pairs(dna_structure,pairs_list):
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
    for nucleotide in nucleotides[1:-1]:
        if not nucleotide.base_pair and nucleotide.in_leading_strand and nucleotide.next_nucleotide and nucleotide.previous_nucleotide:
            if nucleotide.next_nucleotide.base_pair and nucleotide.previous_nucleotide.base_pair:
                pos_pair_nucleotide_1 = nucleotide.next_nucleotide.base_pair.lag_nucl.previous_nucleotide
                pos_pair_nucleotide_2 = nucleotide.previous_nucleotide.base_pair.lag_nucl.next_nucleotide

                if pos_pair_nucleotide_1 == pos_pair_nucleotide_2 and not pos_pair_nucleotide_2.base_pair:
                    new_base_pair = Base_Pair(nucleotide, pos_pair_nucleotide_1,dna_structure=dna_structure)
                    if new_base_pair.pair_name in ('AT', 'CG', 'AU'):

                        new_base_pair.get_pair_params()
                        new_base_pair.update_references()
                        pairs_list.append(new_base_pair)
    
    pairs_list.sort()
    return pairs_list
    
