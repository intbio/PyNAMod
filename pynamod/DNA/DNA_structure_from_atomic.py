from pynamod.DNA.DNA_structure import DNA_Structure
from pynamod.atomic_analysis.nucleotides_parser import get_all_nucleotides, Nucleotide
from pynamod.atomic_analysis.pairs_parser import get_pairs, Base_Pair
from pynamod.geometry.bp_step_geometry import Geometrical_Parameters

import MDAnalysis as mda
from MDAnalysis.topology.guessers import guess_atom_element
import io
import pypdb
import torch


class DNA_Structure_from_Atomic(DNA_Structure):
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
            self.pairs_list = get_pairs(self.nucleotides)
        else:
            self.pairs_list = self.parse_pairs(pairs_in_structure)
            
        ref_frames = torch.stack([pair.Rm for pair in self.pairs_list])
        
        origins = torch.vstack([pair.om for pair in self.pairs_list]).reshape(-1,1,3)
        
        self.geom_params = Geometrical_Parameters(ref_frames=ref_frames,origins=origins)
        self._set_pair_params_list()
        
    