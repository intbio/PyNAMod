import pandas as pd
import numpy as np
import torch
import nglview as nv
from copy import copy

from pynamod.energy.energy_constants import get_consts_olson_98,BDNA_step
from pynamod.protein_analysis.protein import Protein

class DNA_Structure:
    def __init__(self,**kwards):
        for name,value in kwards.items():
            setattr(self,name,value)
    
    def analyze_protein(self,protein_u=None,n_cg_beads=50,ref_index=None):
        self._analyze_protein(protein_u,n_cg_beads,ref_index)
            
    
    def append_structures(self,structures,first_step_params=BDNA_step[6:]):
        return self._append_structures(structures,first_step_params)
        
        
    def move_to_coord_center(self):
        self._move_to_coord_center()
        
        
    def get_dataframe(self):
        return self._get_df()
        
        
    def move_to_cuda(self):
        self.radii.to('cuda')
        self.eps.to('cuda')
        self.charges.to('cuda')
        self.pairs_params.to('cuda')
        self.steps_params.to('cuda')
        self.base_ref_frames.to('cuda')
        for protein in self.proteins:
            protein.cg_radii.to('cuda')
            protein.eps.to('cuda')
            protein.cg_charges.to('cuda')
            protein.cg_beads_pos.to('cuda') 
        
        
    def view_structure(self,prot_color=[0,0,1],dna_color=[0.6,0.6,0.6],dna_pair_r=5):
        return self._view_structure(prot_color,dna_color,dna_pair_r)
        
    def copy(self):
        return self._copy()
        
    def get_proteins_attr(self,attr):
        if self.proteins:
            return torch.cat([getattr(protein,attr) for protein in self.proteins])
        else:
            return torch.empty(0)

    
    def _set_pair_params_list(self):
        self.radii = torch.Tensor([pair.radius for pair in self.pairs_list])
        self.eps = torch.Tensor([pair.eps for pair in self.pairs_list])
        self.charges = torch.Tensor([pair.charge for pair in self.pairs_list])
        
    
    def _view_structure(self,prot_color,dna_color,dna_pair_r):
        view=nv.NGLWidget()
        dna_len = self.origins.shape[0]
        view.shape.add_buffer('sphere',position=self.origins.flatten().tolist(),
                                  color=dna_color*dna_len,radius=[dna_pair_r]*dna_len)
        for protein in self.proteins:
            view.shape.add_buffer('sphere',position=protein.get_true_pos(self).flatten().tolist(),
                                  color=prot_color*protein.n_cg_beads,radius=protein.cg_radii.tolist())

        return view
    
    def _analyze_protein(self,protein_u,n_cg_beads,ref_index):
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
        self._set_pair_params_list()
        self.base_ref_frames = rebuild_by_full_par_frame_numba(self.steps_params)
        
        return self
    
    def _move_to_coord_center(self):
        self.origins -= self.origins[0].clone()
        ref_R = self.ref_frames[0].clone()
        self.ref_frames = torch.matmul(ref_R.T,self.ref_frames)
        self.origins = torch.matmul(self.origins.reshape(-1,1,3),ref_R)
    
    
    def _get_df(self):
        pairs_data = [(pair.lead_nucl.resid,pair.lead_nucl.segid,pair.lead_nucl.restype,
              pair.lag_nucl.restype,pair.lag_nucl.segid,pair.lag_nucl.resid) for pair in self.pairs_list]
        labels = ['resid1','segid1','restype1','restype2','segid2','resid2']
        df = pd.DataFrame(pairs_data,columns = labels)
        labels = ['Shear','Stretch','Stagger','Buckle','Prop-Tw','Opening','Shift','Slide','Rise','Tilt','Roll','Twist']
        params = np.hstack([self.pairs_params,self.steps_params])
        params_df = pd.DataFrame(np.hstack([self.pairs_params,self.steps_params]),columns=labels)
        return pd.concat([df,params_df],axis=1)
    
    def _copy(self):
        new = copy(self)
        new.geom_params = new.geom_params.copy()
        new.radii = new.radii.clone()
        new.eps = new.eps.clone()
        new.charges = new.charges.clone()
        new.proteins = [protein.copy() for protein in new.proteins]
        new.pairs_list = [pair.copy() for pair in new.pairs_list]
        if hasattr(new, 'nucleotides'):
            new.nucleotides = [nucleotide.copy() for nucleotide in new.nucleotides]
    
    @property
    def step_params(self):
        return self.geom_params.local_params
    
    @step_params.setter
    def step_params(self,value):
        self.geom_params.local_params = value
        
    @property
    def ref_frames(self):
        return self.geom_params.ref_frames
    
    @ref_frames.setter
    def ref_frames(self,value):
        self.geom_params.ref_frames = value
        
    @property
    def origins(self):
        return self.geom_params.origins
    
    @origins.setter
    def origins(self,value):
        self.geom_params.origins = value
    def __repr__(self):
        return f'<DNA structure with {len(self.pairs_list)} nucleotide pairs and {len(self.proteins)} proteins>'
    
    
    def __getitem__(self,sl):
        attrs = self.__dict__.copy()
        for attr in ('nucleotides', 'pairs_list', 'geom_params', 'radii', 'eps', 'charges'):
            attrs[attr] = getattr(self,attr)[sl]
        proteins = []
        for protein in self.proteins:
            if protein.ref_pair in attrs['pairs_list']:
                proteins.append(protein)
                
        attrs['proteins'] = proteins
        return DNA_Structure(**attrs)
        
    
        