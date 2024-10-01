import pandas as pd
import numpy as np
import torch
import nglview as nv
from Bio.Seq import Seq

from pynamod.energy_constants import get_consts_olson_98,BDNA_step
from pynamod.Protein_structure_analysis import Protein
from pynamod.bp_step_geometry import rebuild_by_full_par_frame_numba

class DNA_Structure:
    def __init__(self,proteins=[]):
        
        self.pairs_list = []
        self.radii = []
        self.eps = []
        self.charges = []
        self.pairs_params = np.empty((0,6))
        self.steps_params = np.empty((0,6))
        self.proteins = proteins
        
    
    def analyze_protein(self,protein_u=None,n_cg_beads=50,ref_index=None):
        self._analyze_protein(protein_u=protein_u,n_cg_beads=n_cg_beads,ref_index=ref_index)
            
    
    def append_structures(self,structures,first_step_params=BDNA_step[6:]):
        return self._append_structures(structures,first_step_params)
        
        
    def move_to_coord_center(self):
        self.base_ref_frames[:,3,:3] -= self.base_ref_frames[0,3,:3]
        ref_R = self.base_ref_frames[0,:3,:3].copy()
        self.base_ref_frames[:,:3,:3] = np.matmul(ref_R.T,self.base_ref_frames[:,:3,:3])
        self.base_ref_frames[:,3,:3] = np.matmul(self.base_ref_frames[:,3,:3],ref_R)
        
    def move_to_cuda(self):
        self.base_ref_frames = torch.Tensor(self.base_ref_frames).to("cuda")        
        
    def get_proteins_attr(self,attr):
        return [getattr(protein,attr) for protein in self.proteins]
    
    
    def set_pair_params_list(self):
        self.radii = np.array([pair.radius for pair in self.pairs_list])
        self.eps = np.array([pair.eps for pair in self.pairs_list])
        self.charges = np.array([pair.charge for pair in self.pairs_list])
        
        
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
                                  color=prot_color*protein.n_cg_beads,radius=protein.cg_radii.tolist())

        return view
    
    def _analyze_protein(self,protein_u=None,n_cg_beads=50,ref_index=None):
        if not ref_index:
            ref_index = len(self.pairs_list)//2
        if protein_u is None:
            protein_u = self.u.select_atoms('protein')
    
        self.proteins.append(Protein(protein_u,n_cg_beads=n_cg_beads,ref_pair = self.pairs_list[ref_index]))
        self.proteins[-1].build_cg_model()
        
    def _append_structures(self,structures,first_step_params):
        for structure in structures:
            structure.steps_params[0,:] = first_step_params
            self.pairs_params = np.vstack([self.pairs_params,structure.pairs_params])
            self.steps_params = np.vstack([self.steps_params,structure.steps_params])
            self.proteins += structure.proteins
            for pair in structure.pairs_list:
                pair.DNA_structure = self
            self.pairs_list += structure.pairs_list
        self.steps_params[0,:] = 0
        self.set_pair_params_list()
        self.base_ref_frames = rebuild_by_full_par_frame_numba(self.steps_params)
        
        return self
    
        
    
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
        self.base_ref_frames = rebuild_by_full_par_frame_numba(self.steps_params)
        self.set_pair_params_list()
                
        