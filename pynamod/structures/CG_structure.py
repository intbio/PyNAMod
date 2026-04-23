import MDAnalysis as mda
import io
import pypdb
import nglview as nv
import torch
import numpy as np
import matplotlib as mpl
from MDAnalysis.topology.guessers import guess_atom_element
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.analysis import align
import matplotlib.pyplot as plt
from pynamod.structures.DNA_structure import DNA_Structure
from pynamod.structures.rlsp_group import Protein, Real_Space_Beads_Groups
from pynamod.atomic_analysis.nucleotides_parser import check_if_nucleotide
from pynamod.structures.nucleotides_models import Nucleotide_Creator


class CG_Structure:
    '''CG_Structure is one of the main classes of PyNAMod package. It contains DNA structure and proteins structures that are attached to it. It supports analysis and generation, summation, visualization of CG structures. Slices of this class return CG structure with DNA that includes nucleotide pairs with indexes in slice and proteins that have reference pair in sliced structure.

        Key attributes:

        - **dna** - DNA_Structure object.

        - **proteins** - list of Protein objects.

        - **all_coords** - All_Coords object that contains geometrical parameters of pairs steps in DNA structure, reference frames of these steps and origins of beads associated with these steps and of proteins CG beads. This object could also store trajectories of geometrical parameters.

        - **u** - (optional) contains initial mda Universe if given.
    '''

    def __init__(self, dna_structure=None, proteins=None, mdaUniverse=None,
                 pdb_id=None, file=None, all_coords=None, add_proteins=True,
                 nucleotides_model = '1spbp', trajectory=None):
        if mdaUniverse:
            self.u = mdaUniverse
        elif pdb_id:
            self.u = mda.Universe(io.StringIO(pypdb.pdb_client.get_pdb_file(pdb_id)), format='PDB')
        elif file:
            self.u = mda.Universe(file)
        else:
            self.u = None
        if self.u:
            if not hasattr(self.u.atoms,'types'):
                self.u.add_TopologyAttr('elements', [guess_atom_element(name) for name in self.u.atoms.names])

        self.nucleotides_model = nucleotides_model

        self.dna = DNA_Structure(u=self.u)
        self.dna.cg_structure = self

        if proteins:
            self.proteins = proteins
            for protein in proteins:
                protein.cg_structure = self
            self.rlsp_groups = proteins.copy()
        else:
            self.proteins = []

            self.rlsp_groups = []

    def analyze_dna(self,leading_strands=None,pairs_in_structure=None,sel='(type C or type O or type N) and not protein',
                    trajectory=None,overwrite_existing_dna=False,movable=False):
        '''Method that runs analysis of mda Universe and trajectory if given.

            Arguments:

            **leading_strands**: list of segids of leading strands in DNA.

            **pairs_in_structure**: list of nucleotide pairs in correct order that will be used if given instead of automatic determination. Each item of this list should have the following format: resid, segid of nucleotide in leading strand, resid,segid of nucleotide in lagging strand.

            **sel**: selection string for mda Universe to choose atoms which will be included in analysis.

            **movable** - boolean default value to set for each step for Carlo Simulations.
        '''

        if self.dna.pairs_list and not overwrite_existing_dna:
            raise ValueError('DNA was already analyzed for this CG Structure. Use overwrite_existing_dna to proceed anyway.')

        if trajectory is None:
            trajectory = self.u.trajectory[1:]
        self.dna.build_from_u(leading_strands,pairs_in_structure,len(trajectory)+1,sel,overwrite_existing_dna,movable=movable)

        if len(trajectory) != 0:
            self.dna.analyze_trajectory(trajectory)

        self._add_nucleotides()

    def build_dna(self, sequence, movable=True):
        '''Method that runs generation of linear DNA structure with given sequence. Each pair of nucleotides and each step of pairs gains similar average BDNA parameters.

            Arguments:

            **sequence** - string of nucleotide base types to generate structure from.

            **movable** - boolean default value to set for each step for Carlo Simulations.
            '''
        if self.dna.pairs_list:
            raise ValueError('DNA was already initialized for this CG Structure.')

        self.dna.generate(sequence, movable=movable)
        self._add_nucleotides()

    def save_to_h5(self, file, **dataset_kwards):
        self.dna.save_to_h5(file, **dataset_kwards)
        for i, protein in enumerate(self.rlsp_groups):
            protein.save_to_h5(file, group_name=f'protein_{i}_CG_parameters', **dataset_kwards)

    def load_from_h5(self, file):
        self.dna.load_from_h5(file)
        for i in range(len(file)-1):

            if file[f'protein_{i}_CG_parameters']['supdata'][0] > 6:
                protein = Protein(file[f'protein_{i}_CG_parameters']['supdata'][0])
                ref_ind = protein.load_from_h5(file, group_name=f'protein_{i}_CG_parameters')
                protein.ref_pair = self.dna.pairs_list[ref_ind]
                protein.cg_structure = self
                self.rlsp_groups.append(protein)
                self.proteins.append(protein)
            else:
                rlsp = Real_Space_Beads_Groups(file[f'protein_{i}_CG_parameters']['supdata'][0])
                ref_ind = rlsp.load_from_h5(file, group_name=f'protein_{i}_CG_parameters')
                rlsp.ref_pair = self.dna.pairs_list[ref_ind]
                rlsp.cg_structure = self
                self.rlsp_groups.append(rlsp)

        return self

    def analyze_protein(self, protein_u=None, n_cg_beads=50, ref_index=None, binded_dna_len=None):
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
            if hasattr(protein_u, 'altLocs'):
                protein_u = protein_u[protein_u.altLocs == '']

        if binded_dna_len is None:
            binded_dna_len = len(self.dna.pairs_list)

        new = Protein(mdaUniverse = protein_u, n_cg_beads=n_cg_beads,
                      ref_pair=self.dna.pairs_list[ref_index], binded_dna_len=binded_dna_len)
        new.cg_structure = self
        new.build_model(self.dna)
        self.proteins.append(new)
        self.rlsp_groups.append(new)
        self.proteins = sorted(self.proteins, key=lambda p: p.ref_pair.ind)
        self.rlsp_groups = sorted(self.rlsp_groups, key=lambda p: p.ref_pair.ind)

    def get_cg_mda_traj(self, allign_sel='all'):
        '''Method that creates trajectory of CG model as a mda Universe.

            Arguments:

            **allign_sel** - selection for mda Universe. Atoms in this selection will be used to allign model in all frames.
            '''

        n_parts = sum([protein.n_cg_beads for protein in self.rlsp_groups])

        segids = [item for i, group in enumerate(self.rlsp_groups) for item in [i]*group.n_cg_beads]
        resnames = ['prot']*n_parts
        resids = np.arange(n_parts)
        coords = []
        for ts in self.dna.trajectory:
            frame_coord = torch.tensor(self.dna.origins.reshape(1,-1,3))
            if self.proteins:
                frame_coord = torch.vstack([protein.origins for protein in self.proteins]).reshape(1,-1,3)
            coords.append(frame_coord)
        coords = torch.cat(coords).numpy()
        segid_names = ['D'] + [f'P{i}' for i in range(len(self.proteins))]
        u = mda.Universe.empty(n_parts,n_residues=n_parts,n_segments=len(segid_names),
                           atom_resindex=np.arange(n_parts),
                           residue_segindex=segids,
                           trajectory=True)
        u.add_TopologyAttr('name', resnames)
        u.add_TopologyAttr('resname', resnames)
        u.add_TopologyAttr('resid', resids)
        u.add_TopologyAttr('segid', segid_names)
        u.add_TopologyAttr('charge', self.charges)
        u.add_TopologyAttr('radius', self.radii)
        u.load_new(coords, format=MemoryReader)
        alignment = align.AlignTraj(u, u,in_memory=True, select=allign_sel)
        alignment.run()
        return u

    def view_structure(self, disable_charge_bar=True, max_charge=None):
        '''Method for visualization of current CG structure

            Arguments:

            **prot_color**,**dna_color** - lists of rgb format colors with values from 0 to 1 that are used for protein and dna beads respectively.

            **DNA_pair_r** - Radius of DNA bead in visualization.
            '''
        view=nv.NGLWidget()
        dna_len = self.dna.origins.shape[0]
        fig = plt.figure()
        ax = fig.add_axes([0.05, 0.80, 0.9, 0.1])
        all_charges = self.charges
        if not max_charge:
            max_charge = all_charges.abs().max().ceil().item()
        norm = mpl.colors.Normalize(-max_charge, max_charge)
        cb = mpl.colorbar.Colorbar(ax, orientation='horizontal', 
                                       cmap='bwr_r',
                                       norm=norm,
                                       label='charge',
                                       ticks=[-max_charge,0,max_charge])


        colors = np.array([cb.cmap(norm(c))[:3] for c in self.charges]).flatten().tolist()
        view.shape.add_buffer('sphere',position=self.origins.flatten().tolist(),
                              color=colors,radius=self.radii.tolist())

        if disable_charge_bar:
            plt.close(fig)

        return view

    def append_structures(self, structures):
        '''Method for summation of CG structures. It can be used to modify existing structure by adding other structures to it or it can be called on an empty object to keep existing structures unchanged.

            Arguments:

            **structures** - list of CG structures to append.
        '''
        structures = [structure.copy() for structure in structures]
        self.proteins += [protein for structure in structures for protein in structure.proteins]
        self.rlsp_groups += [group for structure in structures for group in structure.rlsp_groups]
        update_value = len(self.dna.pairs_list)
        self.dna.append_structures([structure.dna for structure in structures], copy=False)
        for structure in structures:
            for group in structure.rlsp_groups:
                group.cg_structure = self
                group.ref_pair = self.dna.pairs_list[group.ref_pair.ind+update_value]
            update_value += len(structure.dna.pairs_list)

        return self

    def copy(self):
        '''Method that creates a deep copy of the CG structure.
        '''
        new = CG_Structure(mdaUniverse=self.u)
        new.dna = self.dna.copy()
        new.rlsp_groups = []
        for rlsp in self.rlsp_groups:
            rlsp = rlsp.copy()
            new.rlsp_groups.append(rlsp)
            ind = new.rlsp_groups[-1].ref_pair.ind
            new.rlsp_groups[-1].ref_pair = new.dna.pairs_list[ind]
            rlsp.cg_structure = new
            if rlsp.n_cg_beads > 6:
                new.proteins.append(rlsp)
        return new

    def to(self, device):
        '''Method to send all tensors that are stored in related classes to a given device.

            Arguments:

            **device** - could be 'cuda' or 'cpu' - device to send tensors to.
            '''
        self.all_coords.to(device)
        self.dna.to(device)
        for group in self.rlsp_groups:
            group.to(device)

    def _add_nucleotides(self):
        '''Supported models:
        - 1spbp
        - 1spn
        - 3spn
        '''
        nucleotide_creator = Nucleotide_Creator(self.nucleotides_model)

        for pair in self.dna.pairs_list:
            ind = pair.ind
            pair_beads = nucleotide_creator.create(pair, self.dna.origins[ind], self.dna.ref_frames[ind])
            pair_beads.cg_structure = self
            self.rlsp_groups.append(pair_beads)

        self.rlsp_groups = sorted(self.rlsp_groups, key=lambda p: p.ref_pair.ind)

    def __getitem__(self, sl):
        it = self.copy()
        it.dna.__getitem__(sl)
        rlsp_groups = []
        for group in it.rlsp_groups:
            if group.ref_pair in it.dna.pairs_list:
                rlsp_groups.append(group)
                if sl.step < 0:
                    group.ref_vectors *= -1

        it.rlsp_groups = rlsp_groups

        return it

    @property
    def origins(self):
        return torch.vstack([group.origins for group in self.rlsp_groups])

    @property
    def radii(self):
        return torch.cat([group.radii for group in self.rlsp_groups])

    @property
    def eps(self):
        return torch.cat([group.eps for group in self.rlsp_groups])

    @property
    def charges(self):
        return torch.cat([group.charges for group in self.rlsp_groups])