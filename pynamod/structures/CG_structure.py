import MDAnalysis as mda
import io
import pypdb
import nglview as nv
import torch
import numpy as np
from MDAnalysis.topology.guessers import guess_atom_element
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.analysis import align

from pynamod.structures.DNA_structure import DNA_Structure
from pynamod.structures.protein import Protein


class CG_Structure:
    def __init__(self,dna_structure=None,proteins = None,mdaUniverse = None,pdb_id=None,file=None):
        if mdaUniverse:
            self.u = mdaUniverse
        elif pdb_id:
            self.u = mda.Universe(io.StringIO(pypdb.get_pdb_file(pdb_id)), format='PDB')
        elif file:
            self.u = mda.Universe(file)
        else:
            self.u = None
        if self.u:
            self.u.add_TopologyAttr('elements',[guess_atom_element(name) for name in self.u.atoms.names])
        if dna_structure:
            self.dna = dna_structure
        else:
            try:
                traj_len = len(self.u.trajectory)
            except AttributeError:
                traj_len = 1
            self.dna = DNA_Structure(u=self.u,traj_len=traj_len)
        if proteins:
            self.proteins = proteins
        else:
            self.proteins = []
        
    def analyze_dna(self,leading_strands=None,pairs_in_structure=None,sel=None):
        self.dna.build_from_u(leading_strands,pairs_in_structure,sel=sel)
        
        if len(self.u.trajectory) != 1:
            self.dna.analyze_trajectory(self.u)
            
    def build_dna(self,sequence):
        self.dna.generate(sequence)
        
    def analyze_protein(self,protein_u=None,n_cg_beads=50,ref_index=None):
        if not ref_index:
            ref_index = len(self.dna.pairs_list)//2
        if protein_u is None:
            protein_u = self.u.select_atoms('protein')
            protein_u = protein_u[protein_u.altLocs == '']
    
        self.proteins.append(Protein(protein_u,n_cg_beads=n_cg_beads,ref_pair = self.dna.pairs_list[ref_index]))
        self.proteins[-1].build_model()
        
    def get_cg_mda_traj(self,allign_sel='all'):
        dna_len = len(self.dna.pairs_list)
        prot_len = sum([protein.n_cg_beads for protein in self.proteins])
        n_parts = dna_len + prot_len
        segids = [0]*dna_len + [item for i,protein in enumerate(self.proteins) for item in [1+i]*protein.n_cg_beads]
        resnames = ['dna']*dna_len + ['prot']*prot_len
        resids = np.arange(dna_len + prot_len)
        charges = [pair.charge for pair in self.dna.pairs_list] + list(self.get_proteins_attr('charges'))
        coords = []
        
        for ts in self.dna.trajectory:
            frame_coord = self.dna.origins.reshape(1,-1,3)
            inds = [protein.ref_pair.get_index() for protein in self.proteins]
            prot_coord = torch.cat([protein.get_true_pos(ref_om=self.dna.origins[ind],ref_Rm = self.dna.ref_frames[ind]).reshape(1,-1,3)
                                                     for protein,ind in zip(self.proteins,inds)],dim=1)
            
            frame_coord = torch.cat([frame_coord,prot_coord],dim=1)
            coords.append(frame_coord)
        coords = torch.cat(coords).numpy()
        n_frames = coords.shape[0]
        segid_names = ['D'] + [f'P{i}' for i in range(len(self.proteins))]
        u = mda.Universe.empty(n_parts,n_residues=n_parts,n_segments=len(segid_names),
                           atom_resindex=np.arange(n_parts),
                           residue_segindex=segids,
                           trajectory=True)
        u.add_TopologyAttr('name',resnames)
        u.add_TopologyAttr('resname',resnames)
        u.add_TopologyAttr('resid',resids)
        u.add_TopologyAttr('segid',segid_names)
        u.add_TopologyAttr('charge',charges)
        u.load_new(coords, format=MemoryReader)
        alignment = align.AlignTraj(u, u,in_memory=True,select=allign_sel)
        alignment.run()   
        return u

    
    def view_structure(self,prot_color=[0,0,1],dna_color=[0.6,0.6,0.6],dna_pair_r=5):
        view=nv.NGLWidget()
        dna_len = self.dna.origins.shape[0]
        view.shape.add_buffer('sphere',position=self.dna.origins.flatten().tolist(),
                                  color=dna_color*dna_len,radius=[dna_pair_r]*dna_len)
        for protein in self.proteins:
            view.shape.add_buffer('sphere',position=protein.get_true_pos(self.dna).flatten().tolist(),
                                  color=prot_color*protein.n_cg_beads,radius=protein.radii.tolist())

        return view    

    def append_structures(self,structures):
        structures = [structure.copy() for structure in structures]
        self.dna.append_structures([structure.dna for structure in structures],copy=False)
        for structure in structures:
            for protein in structure.proteins:
                self.proteins.append(protein)
                
        return self
        
        
    def copy(self):
        new = CG_Structure(mdaUniverse=self.u)
        new.dna = self.dna.copy()
        new.proteins = []
        for protein in self.proteins:
            new.proteins.append(protein.copy())
            ind = new.proteins[-1].ref_pair.get_index(self.dna)
            new.proteins[-1].ref_pair = new.dna.pairs_list[ind]
        return new
    
    def get_proteins_attr(self,attr):
        if self.proteins:
            return torch.cat([getattr(protein,attr) for protein in self.proteins])
        else:
            return torch.empty(0)
        
    def to_cuda(self):
        self.dna.to_cuda()
        for protein in self.proteins:
            protein.to_cuda()
        
    
    def __getitem__(self,sl):
        it = self.copy()
        it.dna.__getitem__(sl)
        proteins = []
        for protein in it.proteins:
            if protein.ref_pair in it.dna.pairs_list:
                proteins.append(protein)
                if sl.step < 0:
                    protein.ref_vectors *= -1
                
        it.proteins = proteins
        
        return it
    
                
    @property
    def radii(self):
        return torch.cat([self.dna.radii]+[protein.radii for protein in self.proteins])
    
    @property
    def origins(self):
        return torch.cat([self.dna.origins.reshape(-1,3)]+[protein.origins for protein in self.proteins])
    
    @property
    def eps(self):
        return torch.cat([self.dna.eps]+[protein.eps for protein in self.proteins])
    
    @property
    def charges(self):
        return torch.cat([self.dna.charges]+[protein.charges for protein in self.proteins])