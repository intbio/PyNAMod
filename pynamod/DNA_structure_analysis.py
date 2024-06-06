import pandas as pd
import numpy as np
import io
import pypdb
import nglview as nv
import MDAnalysis as mda
from MDAnalysis.topology.guessers import guess_atom_element
from Bio.Seq import Seq

from pynamod.energy_constants import get_consts_olson_98,BDNA_step
from pynamod.Nucleotides_parser import get_all_nucleotides, Nucleotide
from pynamod.Pairs_parser import get_pairs, get_pairs_and_step_params, Base_pair
from pynamod.Protein_structure_analysis import Protein
from pynamod.bp_step_geometry import rebuild_by_full_par_frame_numba

class DNA_structure:
    def __init__(self,proteins=[]):
        self.pairs_list = []
        
        self.proteins = proteins
        
    
    def analyze_protein(self,protein_u=None,n_cg_beads=50,ref_index=None):
        if not ref_index:
            ref_index = len(self.pairs_list)//2
        if protein_u is None:
            protein_u = self.u.select_atoms('protein')
    
        self.proteins.append(Protein(protein_u,n_cg_beads=n_cg_beads,ref_pair = self.pairs_list[ref_index]))
        self.proteins[-1].build_cg_model()
        
    def get_DataFrame(self):
        pairs_data = [(pair.lead_nucl.resid,pair.lead_nucl.segid,pair.lead_nucl.restype,
              pair.lag_nucl.restype,pair.lag_nucl.segid,pair.lag_nucl.resid) for pair in self.pairs_list]
        labels = ['resid1','segid1','restype1','restype2','segid2','resid2']
        df = pd.DataFrame(pairs_data,columns = labels)
        labels = ['Shear','Stretch','Stagger','Buckle','Prop-Tw','Opening','Shift','Slide','Rise','Tilt','Roll','Twist']
        params = np.hstack([self.pairs_params,self.steps_params])
        params_df = pd.DataFrame(np.hstack([self.pairs_params,self.steps_params]),columns=labels)
        return pd.concat([df,params_df],axis=1)
    
    def view_structure(self,prot_color=[0,0,1],dna_color=[0.6,0.6,0.6],dna_pair_r=5):
        view=nv.NGLWidget()
        dna_len = self.base_ref_frames.shape[0]
        view.shape.add_buffer('sphere',position=self.base_ref_frames[:,3,:3].flatten().tolist(),
                                  color=dna_color*dna_len,radius=[dna_pair_r]*dna_len)
        for protein in self.proteins:
            view.shape.add_buffer('sphere',position=protein.get_true_pos().flatten().tolist(),
                                  color=prot_color*protein.n_cg_beads,radius=protein.cg_radii)

        return view
    
    
    def append_structures(self,structures,first_step_params=BDNA_step[6:]):
        for structure in structures:
            structure.steps_params[0,:] = first_step_params
            self.pairs_params = np.vstack([self.pairs_params,structure.pairs_params])
            self.steps_params = np.vstack([self.steps_params,structure.steps_params])
            self.proteins += structure.proteins
            for pair in structure.pairs_list:
                pair.DNA_structure = self
            self.pairs_list += structure.pairs_list
        self.base_ref_frames = rebuild_by_full_par_frame_numba(np.hstack([self.pairs_params,self.steps_params]))
        
    def move_to_coord_center(self):
        self.base_ref_frames[:,3,:3] -= self.base_ref_frames[0,3,:3]
        ref_R = self.base_ref_frames[0,:3,:3].copy()
        self.base_ref_frames[:,:3,:3] = np.matmul(ref_R.T,self.base_ref_frames[:,:3,:3])
        self.base_ref_frames[:,3,:3] = np.matmul(self.base_ref_frames[:,3,:3],ref_R)
    
class DNA_structure_from_atomic(DNA_structure):
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

        if pairs_in_structure == []:
            get_pairs(self)
        else:
            self.pairs_list = self.parse_pairs(pairs_in_structure)
            
        self.pairs_params = np.zeros((len(self.pairs_list),6))
        self.steps_params = np.zeros((len(self.pairs_list),6))
        self.base_ref_frames = np.zeros((len(self.pairs_list),4,4))
        get_pairs_and_step_params(self) 

    
class DNA_structure_generated(DNA_structure):
    def __init__(self,sequence,proteins=[]):
        super().__init__(proteins=proteins)
        DNA_length = len(sequence)
        self.pairs_params = np.tile(BDNA_step[:6],DNA_length).reshape(-1,6)
        self.steps_params = np.zeros((DNA_length,6))
        
        sequence = sequence.upper()
        averages = get_consts_olson_98()[0]
        seq = 'atcg'*15
        rev_seq = Seq('atcg'*15).reverse_complement()
        prev_lead_nucl = prev_lag_nucl = prev_pair =  None
        for i,(lead_res,lag_res) in enumerate(zip(seq.upper(),rev_seq.upper())):
            
            lead_nucl = Nucleotide(lead_res, i, 'A', True)
            lag_nucl = Nucleotide(lag_res, DNA_length-i, 'B', False)
            pair = Base_pair(lead_nucl,lag_nucl,self)
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
        self.base_ref_frames = rebuild_by_full_par_frame_numba(np.hstack([self.pairs_params,self.steps_params]))

                
        