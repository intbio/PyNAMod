import pandas as pd
import numpy as np
import io
import pypdb
import MDAnalysis as mda
from MDAnalysis.topology.guessers import guess_atom_element

from pynamod.Nucleotides_parser import get_all_nucleotides
from pynamod.Pairs_parser import get_pairs, get_pairs_and_step_params, Base_pair

class DNA_structure:
    def __init__(self,mdaUniverse=None,file=None,pdb_id=None,leading_strands=[]):
            if mdaUniverse:
                self.u = mdaUniverse
            elif pdb_id:
                self.u = mda.Universe(io.StringIO(pypdb.get_pdb_file(pdb_id)), format='PDB')
            elif file:
                self.u = mda.Universe(file)
            self.u.add_TopologyAttr('elements',[guess_atom_element(name) for name in self.u.atoms.names])
            self.leading_strands = leading_strands
            self.proteins = pd.DataFrame(columns=['Segids','of_vec'])
            
    
    
    def parse_pairs(self,pairs_in_structure):
        for pair_data in pairs_in_structure:
            resid1,segid1,resid2,segid2 = pair_data
            nucl1 = nucl2 = None
            for nucl in self.nucleotides:
                if not nucl1 and nucl.resid == resid1 and nucl.segid == segid1:
                    nucl1 = nucl
                elif not nucl2 and nucl.resid == resid2 and nucl.segid == segid2:
                    nucl2 = nucl
            pair = Base_pair(nucl1,nucl2)
            pair.update_references(self.pairs_list)
    
    def analyze_DNA(self,pairs_in_structure=[]):
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
        self.pairs_list = []
        if pairs_in_structure is not None:
            self.parse_pairs(pairs_in_structure)
        else:
            self.get_pairs()
            
        self.pairs_params = np.zeros((len(self.pairs_list),6))
        self.steps_params = np.zeros((len(self.pairs_list),6))
        self.base_ref_frames = np.zeros((len(self.pairs_list),4,4))
        get_pairs_and_step_params(self)
    
    def analyze_protein(self,proteins_segids):
        center_pair = self.pairs_list[len(self.pairs_list)//2]
        o1 = center_pair.pair_om
        ref_mat = center_pair.pair_Rm
        for segid in proteins_segids:
            o2 = self.u.select_atoms(f'protein and segid {segid}').center_of_geometry()
            of_vec = np.matmul(ref_mat,o2) - o1
            self.proteins = pd.concat([self.proteins,pd.DataFrame(((segid,of_vec),),columns=['Segids','of_vec'])])
        
    
    def get_DataFrame(self):
        pairs_data = [(pair.lead_nucl.resid,pair.lead_nucl.segid,pair.lead_nucl.restype,
              pair.lag_nucl.restype,pair.lag_nucl.segid,pair.lag_nucl.resid) for pair in self.pairs_list]
        labels = ['resid1','segid1','restype1','restype2','segid2','resid2']
        df = pd.DataFrame(pairs_data,columns = labels)
        labels = ['Shear','Stretch','Stagger','Buckle','Prop-Tw','Opening','Shift','Slide','Rise','Tilt','Roll','Twist']
        params = np.hstack([self.pairs_params,self.steps_params])
        params_df = pd.DataFrame(np.hstack([self.pairs_params,self.steps_params]),columns=labels)
        return pd.concat([df,params_df],axis=1)