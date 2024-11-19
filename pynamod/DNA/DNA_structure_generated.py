from pynamod.DNA.DNA_structure import DNA_Structure
from pynamod.energy.energy_constants import get_consts_olson_98,BDNA_step
from pynamod.atomic_analysis.nucleotides_parser import Nucleotide
from pynamod.atomic_analysis.pairs_parser import Base_Pair
import numpy as np
from Bio.Seq import Seq


class DNA_Structure_Generated(DNA_Structure):
    def __init__(self,sequence,proteins=[]):
        super().__init__(proteins=proteins)
        DNA_length = len(sequence)
        self.pairs_params = np.tile(BDNA_step[:6],DNA_length).reshape(-1,6)
        self.steps_params = np.zeros((DNA_length,6))
        
        averages = get_consts_olson_98()[0]
        rev_sequence = Seq(sequence).reverse_complement()
        prev_lead_nucl = prev_lag_nucl = prev_pair =  None
        for i,(lead_res,lag_res) in enumerate(zip(sequence.upper(),rev_sequence.upper())):
            
            lead_nucl = Nucleotide(lead_res, i, 'A', True)
            lag_nucl = Nucleotide(lag_res, DNA_length-i, 'B', False)
            pair = Base_Pair(lead_nucl,lag_nucl,self)
            lead_nucl.base_pair = lag_nucl.base_pair = pair
            if prev_lead_nucl:
                self.steps_params[i] = averages[prev_lead_nucl.restype+lead_res]
                lead_nucl.previous_nucleotide = prev_lead_nucl
                lag_nucl.previous_nucleotide = prev_lag_nucl
                prev_lead_nucl.next_nucleotide = lead_nucl
                prev_lag_nucl.next_nucleotide = lag_nucl
                
            
            self.pairs_list.append(pair)
            prev_pair = pair
            prev_lead_nucl = lead_nucl
            prev_lag_nucl = lag_nucl
        self.base_ref_frames = rebuild_by_full_par_frame_numba(self.steps_params)
        self._set_pair_params_list()
                