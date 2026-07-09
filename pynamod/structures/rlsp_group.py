import MDAnalysis as mda
import numpy as np
import h5py
import torch
import subprocess
import tempfile
from sklearn.neighbors import KNeighborsClassifier


def load_rlspg_from_h5(h5_dataset):
    if 'group_type' not in h5_dataset.keys():
        if h5_dataset['supdata'][0] > 6:
            group_type = 'Protein'
        else:
            group_type = 'DNA Pair'
    else:
        group_type = h5_dataset['group_type'].asstr()[()]
    if group_type == 'Protein':
        rlspg = Protein()
    else:
        rlspg = Real_Space_Beads_Groups()

    rlspg.load_from_h5(h5_dataset, group_type=group_type)
    return rlspg


class Real_Space_Beads_Groups:
    def __init__(self, n_cg_beads=None, ref_vectors=None, charges=None, masses=None,
                 radii=None, ref_ind=None, eps=1,
                 binded_dna_len=None, cg_structure=None, group_type = None):
        self.n_cg_beads = n_cg_beads
        self.ref_ind = ref_ind
        self.ref_vectors = ref_vectors
        self.charges = charges
        self.masses = masses
        self.radii = radii
        self.group_type = group_type
        if isinstance(eps, torch.Tensor) or self.n_cg_beads is None:
            self.eps = eps
        else:
            self.eps = torch.ones(n_cg_beads)*eps

        self.binded_dna_len = binded_dna_len
        self.cg_structure = cg_structure

    def save_to_h5(self, file, group_name='rlspg_0_CG_parameters', **dataset_kwards):
        group = file.create_group(group_name)
        group.create_dataset('ref_vectors', data=self.ref_vectors, **dataset_kwards)
        group.create_dataset('charges', data=self.charges, **dataset_kwards)
        group.create_dataset('radii', data=self.radii, **dataset_kwards)
        group.create_dataset('eps', data=self.eps, **dataset_kwards)
        group.create_dataset('masses',data=self.masses, **dataset_kwards)
        group.create_dataset('supdata', data=[self.n_cg_beads, self.ref_ind, self.binded_dna_len], **dataset_kwards)
        dt = h5py.string_dtype(encoding="utf-8")
        group.create_dataset('group_type', data=self.group_type,dtype=dt, **dataset_kwards)

    def load_from_h5(self, h5_dataset ,group_type=None):

        if group_type is not None:
            self.group_type = group_type
        else:
            self.group_type = h5_dataset['group_type'].asstr()[()]
        self.ref_vectors = torch.from_numpy(h5_dataset['ref_vectors'][:]).to(torch.double)
        self.charges = torch.from_numpy(h5_dataset['charges'][:])
        self.radii = torch.from_numpy(h5_dataset['radii'][:])
        self.eps = torch.from_numpy(h5_dataset['eps'][:])
        if 'masses' in h5_dataset.keys():
            self.masses = torch.from_numpy(h5_dataset['masses'][:])
        self.n_cg_beads = int(h5_dataset['supdata'][0])
        self.binded_dna_len = int(h5_dataset['supdata'][2])
        self.ref_ind = int(h5_dataset['supdata'][1])

    def get_true_pos(self, dna_structure=None, ref_om=None, ref_Rm=None):
        if dna_structure is None:
            dna_structure = self.cg_structure.dna

        if ref_om is None:
            ref_om = torch.tensor(dna_structure.origins[self.ref_ind]).to(self.ref_vectors.dtype)
        if ref_Rm is None:
            ref_Rm = torch.tensor(dna_structure.ref_frames[self.ref_ind]).to(self.ref_vectors.dtype)

        return torch.matmul(self.ref_vectors.reshape(-1,1,3).to(ref_om.dtype), ref_Rm.T) + ref_om

    def copy(self,cg_structure=None):
        if cg_structure is None:
            cg_structure = self.cg_structure
        return type(self)(n_cg_beads=self.n_cg_beads, ref_ind=self.ref_ind,
                       eps=self.eps.clone(), ref_vectors=self.ref_vectors.clone(),
                       radii=self.radii.clone(), charges=self.charges.clone(), masses = self.masses.clone(),
                       cg_structure=cg_structure,  binded_dna_len=self.binded_dna_len,group_type=self.group_type)

    def to(self, device):
        self.charges = self.charges.to(device)
        self.radii = self.radii.to(device)
        self.eps = self.eps.to(device)
        self.ref_vectors = self.ref_vectors.to(device)

    def __repr__(self):
        return f'<{self.group_type} with {self.n_cg_beads} CG beads linked to {self.ref_ind}th pair>' 

    @property
    def origins(self):
        return self.get_true_pos()


class Protein(Real_Space_Beads_Groups):
    '''This class contains protein model as coarse grained beads with radii and charges. This class is always related to CG structure that stores positions of CG beads in All_Coords object. Init function of this class requires pdb2pqr class if initialized from mda Universe without charges.'''

    def __init__(self,n_cg_beads=50, mdaUniverse=None,  ref_vectors=None,
                 charges=None, masses=None, radii=None, ref_ind=None,
                 eps=1, binded_dna_len=None, cg_structure=None,group_type=None):

        super().__init__(n_cg_beads, ref_vectors, charges, masses,
                         radii, ref_ind, eps, binded_dna_len, cg_structure,'Protein')

        if mdaUniverse:
            if hasattr(mdaUniverse.atoms, 'charges'):
                self.u = mdaUniverse
            else:
                pdb_temp = tempfile.NamedTemporaryFile(suffix='.pdb')
                pqr_temp = tempfile.NamedTemporaryFile(suffix='.pqr')
                mdaUniverse.atoms.write(pdb_temp.name)
                process = subprocess.run(['pdb2pqr', '--ff=AMBER','--log-level=CRITICAL', pdb_temp.name, pqr_temp.name])
                self.u = mda.Universe(pqr_temp.name)
        else:
            self.u = None

    def build_model(self, dna_structure, random_state=None):
        '''Runs analysis of atomic structure. 

            Attributes:

            **random_state** - currently not supported.'''
        self._get_cg_centers(dna_structure,random_state)
        self._get_cg_params()

    def _get_cg_centers(self, dna_structure, random_state):
        '''Runs optimization algorithm of CG beads positions according to procedure described in https://doi.org/10.1016/j.str.2006.10.003'''
        steps = 200*self.n_cg_beads

        non_H_atoms = self.u.select_atoms('not type H')
        max_coord = np.max(non_H_atoms.positions, axis=0)
        min_coord = np.min(non_H_atoms.positions, axis=0)
        origins = [np.random.uniform(min_v-10,max_v+10,(self.n_cg_beads,1))
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

            origins = origins + eps * np.exp(-k_beads/lmbd).reshape(-1,1) * (ref_atom_r - origins)

        origins = torch.from_numpy(origins.reshape(-1,1,3))
        ref_om = torch.tensor(dna_structure.origins[self.ref_ind]).to(origins.dtype)
        ref_Rm = torch.tensor(dna_structure.ref_frames[self.ref_ind]).to(origins.dtype)
        self.ref_vectors = torch.matmul((origins - ref_om),ref_Rm)

    def _get_cg_params(self):
        '''This method assigns each atom to a CG beads, than the charge of each bead is determined as a sum of charges of its atoms, and radii is defined by radius of gyration of atom groups of this CG bead.'''
        classifier = KNeighborsClassifier(1)
        classifier.fit(self.origins.reshape(-1,3), range(self.n_cg_beads))

        groups = classifier.predict(self.u.atoms.positions)
        radii = np.zeros(self.n_cg_beads)
        self.charges = np.zeros(self.n_cg_beads)
        self.masses = np.zeros(self.n_cg_beads)

        for i in range(self.n_cg_beads):
            sel = self.u.atoms[groups==i]
            radii[i] = sel.radius_of_gyration()
            self.charges[i] = np.sum(sel.charges)
            self.masses[i] = np.sum(sel.masses)

        self.radii = torch.from_numpy(radii + 1)

        self.charges = torch.from_numpy(self.charges)
        self.masses = torch.from_numpy(self.masses)