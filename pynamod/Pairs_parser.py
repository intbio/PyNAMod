import numpy as np
import pickle
from itertools import combinations, pairwise
from scipy.spatial.transform import Rotation as R
from pynamod.bp_step_geometry import get_params_for_single_step_stock

with open('pynamod/classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)


class Base_pair:
    def __init__(self, lead_nucl, lag_nucl):
        if lag_nucl.in_leading_strand:
            lead_nucl, lag_nucl = lag_nucl, lead_nucl

        self.lead_nucl = lead_nucl
        self.lag_nucl = lag_nucl

        self.pair_name = ''.join(sorted((lead_nucl.restype, lag_nucl.restype))).upper()

        self.previous_pair = None
        self.next_pair = None


    def __lt__(self, other):
        return self.lead_nucl.__lt__(other.lead_nucl)
    
    
    def update_references(self,pairs_list):
        self.lead_nucl.base_pair = self.lag_nucl.base_pair = self
        pairs_list.append(self)
    
    def check_if_pair(self):
        if self.pair_name in ('AT', 'CG', 'AU'):
            dist = np.linalg.norm(self.lead_nucl.o - self.lag_nucl.o)
            if dist < 2.2:
                pred_params = R.align_vectors(self.lead_nucl.R, self.lag_nucl.R)[0].as_euler('zyx', degrees=True)
                pred_params[2] += (pred_params[2] < 0) * 360
                pred_params = np.append(pred_params, dist)
                if bool(classifier.predict(pred_params.reshape(1,-1))):
                    return True

        return False


    def get_pair_and_step_params(self,DNA_structure):
        o1, o2 = self.lead_nucl.o, self.lag_nucl.o
        R1, R2 = self.lead_nucl.R, self.lag_nucl.R
        index = DNA_structure.pairs_list.index(self)
        params_and_ref_frame = get_params_for_single_step_stock(o2, o1, R2, R1,pair_params=True)
        DNA_structure.pairs_params[index], DNA_structure.base_ref_frames[index,3,:3], DNA_structure.base_ref_frames[index,:3,:3] = params_and_ref_frame
        self.pair_om, self.pair_Rm = DNA_structure.base_ref_frames[index,3,:3], DNA_structure.base_ref_frames[index,:3,:3]
        self.pair_params = DNA_structure.pairs_params[index]
        if self.previous_pair:
            om_prev, Rm_prev = self.previous_pair.pair_om, self.previous_pair.pair_Rm
            DNA_structure.steps_params[index], self.step_om, self.step_Rm = get_params_for_single_step_stock(
                                                        om_prev, self.pair_om,Rm_prev, self.pair_Rm)
        else:
            self.step_om, self.step_Rm = np.zeros(3),np.identity(3)
        self.step_params = DNA_structure.steps_params[index]


def get_pairs(DNA_structure):
    '''
    Get pairs of nucleotides for multiple structures based on RandomForest classifier algoritm
    '''

    for pair_candidate in combinations(DNA_structure.nucleotides, 2):
        base_pair = Base_pair(*pair_candidate)
        if base_pair.check_if_pair():
            base_pair.update_references(DNA_structure.pairs_list)
            
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
                    new_base_pair = Base_pair(nucleotide, pos_pair_nucleotide_1)
                    if new_base_pair.pair_name in ('AT', 'CG', 'AU'):
                        new_base_pair.update_references(DNA_structure.pairs_list)

    DNA_structure.pairs_list.sort()
    for cur_pair, next_pair in pairwise(DNA_structure.pairs_list):
        cur_pair.next_pair = next_pair
        next_pair.previous_pair = cur_pair
    

def get_pairs_and_step_params(DNA_structure):
    for base_pair in DNA_structure.pairs_list:
        base_pair.get_pair_and_step_params(DNA_structure)
