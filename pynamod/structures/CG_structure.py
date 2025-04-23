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
    '''CG_Structure is one of the main classes of PyNAMod package. It contains DNA structure and proteins structures that are attached to it. It supports analysis and generation, summation, visualization of CG structures. Slices of this class return CG structure with DNA that includes nucleotide pairs with indexes in slice and proteins that have reference pair in sliced structure.
    
        Key attributes:
        
        - **dna** - DNA_Structure object
        
        - **proteins** - list of Protein objects
        
        - **all_coords** - All_Coords object that contains geometrical parameters of pairs steps in DNA structure, reference frames of these steps and origins of beads associated with these steps and of proteins CG beads. This object could also store trajectories of geometrical parameters.
        
        - **u** - (optional) contains initial mda Universe if given
    '''
    def __init__(self,dna_structure=None,proteins = None,mdaUniverse = None,pdb_id=None,file=None,
                 all_coords = None,add_proteins = True,trajectory=None):
        if mdaUniverse:
            self.u = mdaUniverse
        elif pdb_id:
            self.u = mda.Universe(io.StringIO(pypdb.pdb_client.get_pdb_file(pdb_id)), format='PDB')
        elif file:
            self.u = mda.Universe(file)
        else:
            self.u = None
        if self.u:
            self.u.add_TopologyAttr('elements',[guess_atom_element(name) for name in self.u.atoms.names])
            
        if dna_structure:
            self.dna = dna_structure
        else:
            self.dna = DNA_Structure(u=self.u)
        
        
        self.dna.cg_structure = self
        if proteins:
            self.proteins = proteins
            for protein in proteins:
                protein.cg_structure = self
        else:
            self.proteins = []
        
    def analyze_dna(self,leading_strands=None,pairs_in_structure=None,sel=None,trajectory=None):
        '''Method that runs analysis of mda Universe and trajectory if given.
        
            Arguments:
            
            **leading_strands**: list of segids of leading strands in DNA.
            
            **pairs_in_structure**: list of nucleotide pairs that will be used if given instead of automatic determination. Each item of this list should have the following format: resid, segid of nucleotide in leading strand, resid,segid of nucleotide in lagging strand.
            
            **sel**: selection string for mda Universe to choose atoms which will be included in analysis.
        '''
        if trajectory is None:
            trajectory = self.u.trajectory[1:]
        self.dna.build_from_u(leading_strands,pairs_in_structure,len(trajectory)+1,sel)
        
        if len(trajectory) != 1:
            self.dna.analyze_trajectory(trajectory)
            
    def build_dna(self,sequence):
        '''Method that runs generation of linear DNA structure with given sequence. Each pair of nucleotides and each step of pairs gains similar average BDNA parameters.
            
            Arguments:
            
            **sequence** - string of nucleotide base types to generate structure from.
            '''
        self.dna.generate(sequence)
        
    def save_to_h5(self,file,**dataset_kwards):
        self.dna.save_to_h5(file,**dataset_kwards)
        for i,protein in enumerate(self.proteins):
            protein.save_to_h5(file,group_name=f'protein_{i}_CG_parameters',**dataset_kwards)
            
    def load_from_h5(self,file):
        self.dna.load_from_h5(file)
        i = 0
        while True:
            try:
                protein = Protein()
                ref_ind = protein.load_from_h5(file,group_name=f'protein_{i}_CG_parameters')
                protein.ref_pair = self.dna.pairs_list[ref_ind]
                protein.cg_structure = self
                self.proteins.append(protein)
                i += 1
            except KeyError:
                break
    def analyze_protein(self,protein_u=None,n_cg_beads=50,ref_index=None):
        '''Method that finds protein structure in given mda Universe or Universe attached to the instance of class. Coarse Grained structure is than constructed based on protein and added to the proteins list.
        
            Arguments:
            
            **protein_u** - mda Universe that contains protein to analyze. If None, protein is selected in the Universe stored in the instance of the class.
            
            **n_cg_beads** - number of coarse grained beads in created protein model.
            
            **ref_index** - index of nucleotide pair relative to which position of protein is defined. Note that selection of reference pair is important for slicing, as all proteins without reference pair after slice are dropped from the sliced structure.
            '''
        if not ref_index:
            ref_index = len(self.dna.pairs_list)//2
        if protein_u is None:
            protein_u = self.u.select_atoms('protein')
            protein_u = protein_u[protein_u.altLocs == '']
        
        new = Protein(protein_u,n_cg_beads=n_cg_beads,ref_pair = self.dna.pairs_list[ref_index])
        new.cg_structure = self
        new.build_model()
        self.proteins.append(new)
        
        
    def get_cg_mda_traj(self,allign_sel='all'):
        '''Method that creates trajectory of CG model as a mda Universe.
        
            Arguments:
            
            **allign_sel** - selection for mda Universe. Atoms in this selection will be used to allign model in all frames.
            '''
        dna_len = len(self.dna.pairs_list)
        if self.proteins:
            prot_len = sum([protein.n_cg_beads for protein in self.proteins])
        else:
            prot_len = 0
        n_parts = dna_len + prot_len
        segids = [0]*dna_len + [item for i,protein in enumerate(self.proteins) for item in [1+i]*protein.n_cg_beads]
        resnames = ['dna']*dna_len + ['prot']*prot_len
        resids = np.arange(dna_len + prot_len)
        charges = [pair.charge for pair in self.dna.pairs_list] + list(self.get_proteins_attr('charges'))
        coords = []
        
        for ts in self.dna.trajectory:
            frame_coord = torch.tensor(self.dna.origins.reshape(1,-1,3))
            if self.proteins:
                prot_coord = torch.vstack([protein.origins for protein in self.proteins]).reshape(1,-1,3)
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

    
    def view_structure(self,prot_color=[0,0,1],dna_color=[0.6,0.6,0.6]):
        '''Method for visualization of current CG structure
            
            Arguments:
            
            **prot_color**,**dna_color** - lists of rgb format colors with values from 0 to 1 that are used for protein and dna beads respectively.
            
            **DNA_pair_r** - Radius of DNA bead in visualization.
            '''
        view=nv.NGLWidget()
        dna_len = self.dna.origins.shape[0]
        view.shape.add_buffer('sphere',position=self.dna.origins.flatten().tolist(),
                                  color=dna_color*dna_len,radius=self.dna.pairs_list.radii)
        for protein in self.proteins:
            view.shape.add_buffer('sphere',position=protein.origins.flatten().tolist(),
                                  color=prot_color*protein.n_cg_beads,radius=protein.radii.tolist())

        return view    

    def append_structures(self,structures):
        '''Method for summation of CG structures. It can be used to modify existing structure by adding other structures to it or it can be called on an empty object to keep existing structures unchanged.
        
            Arguments:
            
            **structures** - list of CG structures to append.
        '''
        structures = [structure.copy() for structure in structures]
        self.proteins += [protein for structure in structures for protein in structure.proteins]
        update_value = len(self.dna.pairs_list)
        self.dna.append_structures([structure.dna for structure in structures],copy=False)
        for structure in structures:
            for protein in structure.proteins:
                protein.cg_structure = self
                protein.ref_pair = self.dna.pairs_list[protein.ref_pair.ind+update_value]
            update_value += len(structure.dna.pairs_list)
                
        return self
        
        
    def copy(self):
        '''Method that creates a deep copy of the CG structure.
        '''
        new = CG_Structure(mdaUniverse=self.u)
        new.dna = self.dna.copy()
        new.proteins = []
        for protein in self.proteins:
            new.proteins.append(protein.copy())
            ind = new.proteins[-1].ref_pair.ind
            new.proteins[-1].ref_pair = new.dna.pairs_list[ind]
        return new
    
    def get_proteins_attr(self,attr):
        if self.proteins:
            return torch.cat([getattr(protein,attr) for protein in self.proteins])
        else:
            return torch.empty(0)
        
    def to(self,device):
        '''Method to send all tensors that are stored in related classes to a given device.
        
            Arguments:
            
            **device** - could be 'cuda' or 'cpu' - device to send tensors to.
            '''
        self.all_coords.to(device)
        self.dna.to(device)
        for protein in self.proteins:
            protein.to(device)
        
    
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
        return torch.cat([self.dna.radii.reshape(-1)]+[protein.radii for protein in self.proteins[::-1]])
    
    @property
    def eps(self):
        return torch.cat([self.dna.eps.reshape(-1)]+[protein.eps for protein in self.proteins[::-1]])
    
    @property
    def charges(self):
        return torch.cat([self.dna.charges.reshape(-1)]+[protein.charges for protein in self.proteins[::-1]])