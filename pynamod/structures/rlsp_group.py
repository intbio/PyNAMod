import io
import shutil
import subprocess
import sys
import tempfile

import MDAnalysis as mda
import numpy as np
<<<<<<< HEAD:pynamod/structures/rlsp_group.py
import torch
import subprocess
import tempfile
from sklearn.neighbors import KNeighborsClassifier
=======
import pypdb
import torch
from sklearn.neighbors import NearestNeighbors

>>>>>>> 4c2722a3fcadace9ef261cd86d95e432a14f8376:pynamod/structures/protein.py


class Real_Space_Beads_Groups:
<<<<<<< HEAD:pynamod/structures/rlsp_group.py
    def __init__(self, n_cg_beads=None, ref_vectors=None, charges=None, masses=None,
                 radii=None, ref_pair=None, eps=1,
                 binded_dna_len=None, cg_structure=None):
        self.n_cg_beads = n_cg_beads
        self.ref_pair = ref_pair
        self.ref_vectors = ref_vectors
        self.charges = charges
        self.masses = masses
        self.radii = radii
        if isinstance(eps, torch.Tensor) or self.n_cg_beads is None:
=======
    def __init__(self, ref_pair=None, eps=1, binded_dna_len=None, cg_structure=None):
        self.ref_pair = ref_pair
        if isinstance(eps, torch.Tensor):
>>>>>>> 4c2722a3fcadace9ef261cd86d95e432a14f8376:pynamod/structures/protein.py
            self.eps = eps
        else:
            self.eps = torch.ones(n_cg_beads)*eps

        self.binded_dna_len = binded_dna_len
        self.cg_structure = cg_structure
<<<<<<< HEAD:pynamod/structures/rlsp_group.py
=======

>>>>>>> 4c2722a3fcadace9ef261cd86d95e432a14f8376:pynamod/structures/protein.py

    def save_to_h5(self, file, group_name='rlsp_0_CG_parameters', **dataset_kwards):
        group = file.create_group(group_name)
        group.create_dataset('ref_vectors', data=self.ref_vectors, **dataset_kwards)
        group.create_dataset('charges', data=self.charges, **dataset_kwards)
        group.create_dataset('radii', data=self.radii, **dataset_kwards)
        group.create_dataset('eps', data=self.eps, **dataset_kwards)
        group.create_dataset('supdata', data=[self.n_cg_beads, self.ref_pair.ind, self.binded_dna_len], **dataset_kwards)

    def load_from_h5(self,file, group_name='rlsp_0_CG_parameters'):
                
        self.ref_vectors = torch.from_numpy(file[group_name]['ref_vectors'][:]).to(torch.double)
        self.charges = torch.from_numpy(file[group_name]['charges'][:])
        self.radii = torch.from_numpy(file[group_name]['radii'][:])
        self.eps = torch.from_numpy(file[group_name]['eps'][:])
        self.n_cg_beads = int(file[group_name]['supdata'][0])
        self.binded_dna_len = int(file[group_name]['supdata'][2])
        return int(file[group_name]['supdata'][1])

    def get_true_pos(self, dna_structure=None, ref_om=None, ref_Rm=None):
        if dna_structure is None:
            dna_structure = self.cg_structure.dna

        if ref_om is None:
            ref_om = torch.tensor(dna_structure.origins[self.ref_pair.ind]).to(self.ref_vectors.dtype)
        if ref_Rm is None:
            ref_Rm = torch.tensor(dna_structure.ref_frames[self.ref_pair.ind]).to(self.ref_vectors.dtype)

        return torch.matmul(self.ref_vectors.reshape(-1,1,3).to(ref_om.dtype), ref_Rm.T) + ref_om

    def copy(self):
        return Real_Space_Beads_Groups(n_cg_beads=self.n_cg_beads, ref_pair=self.ref_pair,
                       eps=self.eps.clone(), ref_vectors=self.ref_vectors.clone(),
                       radii=self.radii.clone(), charges=self.charges.clone(),
                       cg_structure=self.cg_structure,  binded_dna_len=self.binded_dna_len)

    def to(self, device):
        self.charges = self.charges.to(device)
        self.radii = self.radii.to(device)
        self.eps = self.eps.to(device)
        self.ref_vectors = self.ref_vectors.to(device)

    def __repr__(self):
        return f'<Real Space Beads group with {self.n_cg_beads} CG beads and linked to {self.ref_pair.ind}th pair>' 
    
    @property
    def origins(self):
        return self.get_true_pos()


class Protein(Real_Space_Beads_Groups):
    '''This class contains protein model as coarse grained beads with radii and charges. This class is always related to CG structure that stores positions of CG beads in All_Coords object. Init function of this class requires pdb2pqr class if initialized from mda Universe without charges.'''

<<<<<<< HEAD:pynamod/structures/rlsp_group.py
    def __init__(self,n_cg_beads=50, mdaUniverse=None,  ref_vectors=None,
                 charges=None, masses=None, radii=None, ref_pair=None,
                 eps=1, binded_dna_len=None, cg_structure=None):

        super().__init__(n_cg_beads, ref_vectors, charges, masses,
                         radii, ref_pair, eps, binded_dna_len, cg_structure)
=======
    def __init__(self, mdaUniverse=None, n_cg_beads=50, ref_pair=None, eps=1, binded_dna_len=None, cg_structure=None, **kwards):
        self.n_cg_beads = n_cg_beads
>>>>>>> 4c2722a3fcadace9ef261cd86d95e432a14f8376:pynamod/structures/protein.py

        if mdaUniverse:
            if hasattr(mdaUniverse.atoms, 'charges'):
                self.u = mdaUniverse
            else:
                pdb_temp = tempfile.NamedTemporaryFile(suffix='.pdb')
                pqr_temp = tempfile.NamedTemporaryFile(suffix='.pqr')
                # PDBWriter warns if formalcharges is missing; PDBs rarely carry that column.
                if not hasattr(mdaUniverse.atoms, 'formalcharges'):
                    mdaUniverse.universe.add_TopologyAttr('formalcharges')
                mdaUniverse.atoms.write(pdb_temp.name)
                cmd = ['pdb2pqr', '--ff=AMBER', '--log-level=CRITICAL', pdb_temp.name, pqr_temp.name]
                if shutil.which('pdb2pqr') is None:
                    cmd = [sys.executable, '-m', 'pdb2pqr', '--ff=AMBER', '--log-level=CRITICAL', pdb_temp.name, pqr_temp.name]
                process = subprocess.run(cmd)
                self.u = mda.Universe(pqr_temp.name)
        else:
            self.u = None
<<<<<<< HEAD:pynamod/structures/rlsp_group.py

    def build_model(self, dna_structure, random_state=None):
        '''Runs analysis of atomic structure. 
=======
        self.ref_pair = ref_pair
        if isinstance(eps, torch.Tensor):
            self.eps = eps
        else:
            self.eps = torch.ones(n_cg_beads)*eps

        self.binded_dna_len = binded_dna_len
        self.cg_structure = cg_structure

        for name, value in kwards.items():
            setattr(self, name, value)

    def build_model(self, dna_structure, random_state=None):
        '''Runs analysis of atomic structure.
>>>>>>> 4c2722a3fcadace9ef261cd86d95e432a14f8376:pynamod/structures/protein.py

            Attributes:

            **random_state** - currently not supported.'''
        self._get_cg_centers(dna_structure, random_state)
        self._get_cg_params()
<<<<<<< HEAD:pynamod/structures/rlsp_group.py

=======

    def save_to_h5(self, file, group_name='protein_0_CG_parameters', **dataset_kwards):
        group = file.create_group(group_name)
        group.create_dataset('ref_vectors', data=self.ref_vectors, **dataset_kwards)
        group.create_dataset('charges', data=self.charges, **dataset_kwards)
        group.create_dataset('radii', data=self.radii, **dataset_kwards)
        group.create_dataset('masses', data=self.masses, **dataset_kwards)
        group.create_dataset('eps', data=self.eps, **dataset_kwards)
        group.create_dataset('supdata', data=[self.n_cg_beads, self.ref_pair.ind, self.binded_dna_len], **dataset_kwards)

    def load_from_h5(self, file, group_name='protein_0_CG_parameters'):
        self.ref_vectors = torch.from_numpy(file[group_name]['ref_vectors'][:]).to(torch.double)
        self.charges = torch.from_numpy(file[group_name]['charges'][:])
        self.radii = torch.from_numpy(file[group_name]['radii'][:])
        self.masses = torch.from_numpy(file[group_name]['masses'][:])
        self.eps = torch.from_numpy(file[group_name]['eps'][:])
        self.n_cg_beads = int(file[group_name]['supdata'][0])
        self.binded_dna_len = int(file[group_name]['supdata'][2])
        return int(file[group_name]['supdata'][1])

    def get_true_pos(self, dna_structure=None, ref_om=None, ref_Rm=None):
        if ref_om is None:
            ref_om = dna_structure.origins[self.ref_pair.ind].detach().clone().to(self.ref_vectors.dtype)
            ref_Rm = dna_structure.ref_frames[self.ref_pair.ind].detach().clone().to(self.ref_vectors.dtype)

        return torch.matmul(self.ref_vectors.reshape(-1, 1, 3), ref_Rm.T) + ref_om

    def copy(self):
        return Protein(mdaUniverse=self.u, n_cg_beads=self.n_cg_beads, ref_pair=self.ref_pair,
                       eps=self.eps.clone(), ref_vectors=self.ref_vectors.clone(),
                       radii=self.radii.clone(), charges=self.charges.clone(), masses=self.masses.clone(), cg_structure=self.cg_structure,
                       binded_dna_len=self.binded_dna_len)

    def to(self, device):
        self.charges = self.charges.to(device)
        self.radii = self.radii.to(device)
        self.eps = self.eps.to(device)
        self.ref_vectors = self.ref_vectors.to(device)

>>>>>>> 4c2722a3fcadace9ef261cd86d95e432a14f8376:pynamod/structures/protein.py
    def _get_cg_centers(self, dna_structure, random_state):
        '''Runs optimization algorithm of CG beads positions according to procedure described in https://doi.org/10.1016/j.str.2006.10.003'''
        steps = 200*self.n_cg_beads

        non_H_atoms = self.u.select_atoms('not type H')
        max_coord = np.max(non_H_atoms.positions, axis=0)
        min_coord = np.min(non_H_atoms.positions, axis=0)
<<<<<<< HEAD:pynamod/structures/rlsp_group.py
        origins = [np.random.uniform(min_v-10,max_v+10,(self.n_cg_beads,1))
=======
        origins = [np.random.uniform(min_v-10, max_v+10, (self.n_cg_beads, 1))
>>>>>>> 4c2722a3fcadace9ef261cd86d95e432a14f8376:pynamod/structures/protein.py
                   for min_v, max_v in zip(min_coord, max_coord)]

        origins = np.hstack(origins)

        lmbd_0, eps_0 = 0.2*self.n_cg_beads, 0.3
        lmbd_s, eps_s = 0.01, 0.05

        atoms_number = len(non_H_atoms)
        for step in range(1, steps+1):
            rand = np.random.randint(0, atoms_number)
            ref_atom_r = non_H_atoms[rand].position
            dist = np.linalg.norm(ref_atom_r - origins, axis=1)
            k_beads = np.zeros(dist.shape)
            k_beads[np.argsort(dist)] = np.arange(self.n_cg_beads)

            lmbd = lmbd_0 * (lmbd_s/lmbd_0) ** (step/steps)
            eps = eps_0 * (eps_s/eps_0) ** (step/steps)

<<<<<<< HEAD:pynamod/structures/rlsp_group.py
            origins = origins + eps * np.exp(-k_beads/lmbd).reshape(-1,1) * (ref_atom_r - origins)

        origins = torch.from_numpy(origins.reshape(-1,1,3))
        ref_om = torch.tensor(dna_structure.origins[self.ref_pair.ind]).to(origins.dtype)
        ref_Rm = torch.tensor(dna_structure.ref_frames[self.ref_pair.ind]).to(origins.dtype)
        self.ref_vectors = torch.matmul((origins - ref_om),ref_Rm)

    def _get_cg_params(self):
        '''This method assigns each atom to a CG beads, than the charge of each bead is determined as a sum of charges of its atoms, and radii is defined by radius of gyration of atom groups of this CG bead.'''
        classifier = KNeighborsClassifier(1)
        classifier.fit(self.origins.reshape(-1,3), range(self.n_cg_beads))

        groups = classifier.predict(self.u.atoms.positions)
=======
            origins = origins + eps * np.exp(-k_beads/lmbd).reshape(-1, 1) * (ref_atom_r - origins)

        origins = torch.from_numpy(origins.reshape(-1, 1, 3))
        ref_om = dna_structure.origins[self.ref_pair.ind].detach().clone().to(origins.dtype)
        ref_Rm = dna_structure.ref_frames[self.ref_pair.ind].detach().clone().to(origins.dtype)
        self.ref_vectors = torch.matmul((origins - ref_om), ref_Rm)

    def _get_cg_params(self):
        '''This method assigns each atom to a CG beads, than the charge of each bead is determined as a sum of charges of its atoms, and radii is defined by radius of gyration of atom groups of this CG bead.'''
        centers = self.origins.reshape(-1, 3).detach().cpu().numpy()
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(centers)
        _, groups = nn.kneighbors(self.u.atoms.positions, return_distance=True)
        groups = groups.ravel()
>>>>>>> 4c2722a3fcadace9ef261cd86d95e432a14f8376:pynamod/structures/protein.py
        radii = np.zeros(self.n_cg_beads)
        self.charges = np.zeros(self.n_cg_beads)
        self.masses = np.zeros(self.n_cg_beads)

        for i in range(self.n_cg_beads):
            sel = self.u.atoms[groups == i]
            radii[i] = sel.radius_of_gyration()
            self.charges[i] = np.sum(sel.charges)
            self.masses[i] = np.sum(sel.masses)

        self.radii = torch.from_numpy(radii + 1)

        self.charges = torch.from_numpy(self.charges)
        self.masses = torch.from_numpy(self.masses)

<<<<<<< HEAD:pynamod/structures/rlsp_group.py
    def copy(self):
        return Protein(mdaUniverse=self.u, n_cg_beads=self.n_cg_beads, ref_pair=self.ref_pair,
                       eps=self.eps.clone(), ref_vectors=self.ref_vectors.clone(),
                       radii=self.radii.clone(), charges=self.charges.clone(),
                       cg_structure=self.cg_structure,  binded_dna_len=self.binded_dna_len)

    def __repr__(self):
        return f'<Protein with {self.n_cg_beads} CG beads and linked to {self.ref_pair.ind}th pair>' 
=======
    def __repr__(self):
        return f'<Protein with {self.n_cg_beads} CG beads and linked to {self.ref_pair.ind}th pair>'

    @property
    def origins(self):
        return self.get_true_pos(self.cg_structure.dna)
>>>>>>> 4c2722a3fcadace9ef261cd86d95e432a14f8376:pynamod/structures/protein.py
