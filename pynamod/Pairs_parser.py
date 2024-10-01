import numpy as np
import pickle
from itertools import combinations
from more_itertools import pairwise
from scipy.spatial.transform import Rotation as R
from pynamod.bp_step_geometry import get_params_for_single_step_stock,get_ori_and_mat_from_step_opt,rebuild_by_full_par_frame_numba

with open('pynamod/classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)


class Base_pair:
    def __init__(self, lead_nucl, lag_nucl,DNA_structure,radius=2,charge=-2,eps=0.5):
        if lag_nucl.in_leading_strand:
            lead_nucl, lag_nucl = lag_nucl, lead_nucl

        self.lead_nucl = lead_nucl
        self.lag_nucl = lag_nucl
        self.DNA_structure = DNA_structure
        self.radius = radius
        self.charge = charge
        self.eps = eps

        self.pair_name = ''.join(sorted((lead_nucl.restype, lag_nucl.restype))).upper()

        
    def __lt__(self, other):
        return self.lead_nucl.__lt__(other.lead_nucl)
    
    
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
                    self.DNA_structure.pairs_list.append(self)


    def get_pair_and_step_params(self):
        o1, o2 = self.lead_nucl.o, self.lag_nucl.o
        R1, R2 = self.lead_nucl.R, self.lag_nucl.R
        self.pair_params, self.om, self.Rm = get_params_for_single_step_stock(o2, o1, R2, R1,pair_params=True)
        
        previous_pair = self.get_previous_pair()
        if previous_pair:
            om_prev, Rm_prev = previous_pair.om, previous_pair.Rm
            self.step_params = get_params_for_single_step_stock(om_prev, self.om,Rm_prev, self.Rm)[0]
    

    def get_index(self):
        return self.DNA_structure.pairs_list.index(self) 
    
    def get_previous_pair(self):
        index = self.get_index()
        if index == 0:
            return None
        else:
            return self.DNA_structure.pairs_list[index-1]

    def get_params(self,slices,attr):
        return getattr(self.DNA_structure,attr)[self.get_index()][slices]
    
    def set_params(self,value,slices,attr):
        getattr(self.DNA_structure,attr)[self.get_index()][slices] = value
    
    def set_step_params(self,value,slices,attr):
        self.set_params(value,slices=slices,attr=attr)
        index = self.get_index() - 1
        
        self.DNA_structure.base_ref_frames[index:] = rebuild_by_full_par_frame_numba(self.DNA_structure.steps_params[index:],
                                                                            self.DNA_structure.base_ref_frames[index])
        
    def set_property(slices,attr):
        getter = lambda self: self.get_params(slices=slices,attr=attr)
        
        if attr == 'steps_params':
            setter = lambda self,value: self.set_step_params(value,slices=slices,attr=attr)
            
        else:
            setter = lambda self,value: self.set_params(value,slices=slices,attr=attr)
            
        return property(getter,setter)
    
    pair_params = set_property(np.s_[:],'pairs_params')
    step_params = set_property(np.s_[:],'steps_params')
    Rm = set_property(np.s_[:3,:3],'base_ref_frames')
    om = set_property(np.s_[3,:3],'base_ref_frames')
    
    
def get_pairs(DNA_structure):
    '''
    Get pairs of nucleotides for multiple structures based on RandomForest classifier algoritm
    '''

    for pair_candidate in combinations(DNA_structure.nucleotides, 2):
        
        base_pair = Base_pair(*pair_candidate,DNA_structure)
        base_pair.check_if_pair()
    
    fix_missing_pairs(DNA_structure)

    
def fix_missing_pairs(DNA_structure):
    '''
Add missing pairs by context after classifier algorithm.
Fix the cases
A - T
G   C
T - A
by assuming the middle nucleotides are a pair.
-----
    '''
    for nucleotide in DNA_structure.nucleotides[1:-1]:
        if not nucleotide.base_pair and nucleotide.in_leading_strand and nucleotide.next_nucleotide and nucleotide.previous_nucleotide:
            if nucleotide.next_nucleotide.base_pair and nucleotide.previous_nucleotide.base_pair:
                pos_pair_nucleotide_1 = nucleotide.next_nucleotide.base_pair.lag_nucl.previous_nucleotide
                pos_pair_nucleotide_2 = nucleotide.previous_nucleotide.base_pair.lag_nucl.next_nucleotide

                if pos_pair_nucleotide_1 == pos_pair_nucleotide_2 and not pos_pair_nucleotide_2.base_pair:
                    new_base_pair = Base_pair(nucleotide, pos_pair_nucleotide_1,DNA_structure)
                    if new_base_pair.pair_name in ('AT', 'CG', 'AU'):
                        new_base_pair.update_references()
                        DNA_structure.pairs_list.append(new_base_pair)

    DNA_structure.pairs_list.sort()
    
def get_pairs_and_step_params(DNA_structure):
    for base_pair in DNA_structure.pairs_list:
        base_pair.get_pair_and_step_params()
