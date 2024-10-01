from pynamod.DNA_structure import DNA_structure
from pynamod.Nucleotides_parser import get_all_nucleotides, Nucleotide
from pynamod.Pairs_parser import get_pairs, get_pairs_and_step_params, Base_pair

import MDAnalysis as mda
from MDAnalysis.topology.guessers import guess_atom_element
import io
import pypdb
import numpy as np


class DNA_Structure_from_Atomic(DNA_structure):
    def __init__(self,mdaUniverse=None,file=None,pdb_id=None,leading_strands=[],proteins=[]):
        super().__init__(proteins=proteins)
        if mdaUniverse:
            self.u = mdaUniverse
        elif pdb_id:
            self.u = mda.Universe(io.StringIO(pypdb.get_pdb_file(pdb_id)), format='PDB')
        elif file:
            self.u = mda.Universe(file)
        self.leading_strands = leading_strands
        self.u.add_TopologyAttr('elements',[guess_atom_element(name) for name in self.u.atoms.names])
        
        
    def parse_pairs(self,pairs_in_structure):
        self.pairs_list = []
        for pair_data in pairs_in_structure:
            resid1,segid1,resid2,segid2 = pair_data
            nucl1 = nucl2 = None
            for nucl in self.nucleotides:
                if not nucl1 and nucl.resid == resid1 and nucl.segid == segid1:
                    nucl1 = nucl
                elif not nucl2 and nucl.resid == resid2 and nucl.segid == segid2:
                    nucl2 = nucl
            pair = Base_pair(nucl1,nucl2,self)
            pair.update_references()
    
    
    def analyze_DNA(self,pairs_in_structure=[],dna_eps=0.5):
        '''
    Full analysis of dna in pdb structure. The function is built to be similar to 3dna algorithm(http://nar.oxfordjournals.org/content/31/17/5108.full).
    -----
    input:
        mdaUniverse, file, pypdb_id - PDB structure as a mda Universe object, file path or pdb_id respectively
        leading_strands - strands that will be used to set order of parameters calculations
        pairs_list - list of pairs that will be used instead of classifier algorithm to generate pairs DataFrame. Each element of it should be a tuple of the segid and resid of the first nucleotide in pair and then the segid and resid of the second.
    ----
    returns:
        params_df - pandas DataFrame with calculated intra and inter geometrical parameters and resid, segid and nucleotide type of each nucleotides in pair.
        '''

        self.nucleotides = get_all_nucleotides(self)

        if pairs_in_structure == []:
            get_pairs(self)
        else:
            self.pairs_list = self.parse_pairs(pairs_in_structure)
            
        self.pairs_params = np.zeros((len(self.pairs_list),6))
        self.steps_params = np.zeros((len(self.pairs_list),6))
        self.base_ref_frames = np.zeros((len(self.pairs_list),4,4))
        self.set_pair_params_list()
        get_pairs_and_step_params(self) 