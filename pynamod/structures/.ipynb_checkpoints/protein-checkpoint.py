import MDAnalysis as mda
import numpy as np
import torch
import io
import subprocess
import tempfile
import pypdb
from sklearn.neighbors import KNeighborsClassifier


class Protein:
    '''This class contains protein model as coarse grained beads with radii and charges. This class is always related to CG structure that stores positions of CG beads in All_Coords object. Init function of this class requires pdb2pqr class if initialized from mda Universe without charges.'''
    def __init__(self,mdaUniverse=None,n_cg_beads=50,ref_pair = None,eps=1,**kwards):
        self.n_cg_beads = n_cg_beads
        
        if mdaUniverse:
            if hasattr(mdaUniverse.atoms,'charges'):
                self.u = mdaUniverse
            else:
                pdb_temp = tempfile.NamedTemporaryFile(suffix='.pdb')
                pqr_temp = tempfile.NamedTemporaryFile(suffix='.pqr')
                mdaUniverse.atoms.write(pdb_temp.name)
                process = subprocess.run(['pdb2pqr', '--ff=AMBER','--log-level=CRITICAL', pdb_temp.name, pqr_temp.name])
                self.u = mda.Universe(pqr_temp.name)
        else:
            self.u = None
        self.ref_pair = ref_pair
        if isinstance(eps,torch.Tensor):
            self.eps = eps
        else:
            self.eps = torch.ones(n_cg_beads)*eps
        
        for name,value in kwards.items():
            setattr(self,name,value)
        
    
    def build_model(self,random_state=None):
        '''Runs analysis of atomic structure. 
        
            Attributes:

            **random_state** - currently not supported.'''
        self._get_cg_centers(random_state)
        self._get_cg_params()
        
    def save_to_h5(self,file,group_name='protein_0_CG_parameters',**dataset_kwards):
        group = file.create_group(group_name)
        group.create_dataset('ref_vectors',data=self.ref_vectors,**dataset_kwards)
        group.create_dataset('charges',data=self.charges,**dataset_kwards)
        group.create_dataset('radii',data=self.radii,**dataset_kwards)
        group.create_dataset('masses',data=self.masses,**dataset_kwards)
        group.create_dataset('eps',data=self.eps,**dataset_kwards)
        group.create_dataset('supdata',data=[self.n_cg_beads,self.ref_pair.ind,self.binded_dna_len],**dataset_kwards)
        
    def load_from_h5(self,file,group_name='protein_0_CG_parameters'):
        self.ref_vectors = torch.from_numpy(file[group_name]['ref_vectors'][:]).to(torch.double)
        self.charges = torch.from_numpy(file[group_name]['charges'][:])
        self.radii = torch.from_numpy(file[group_name]['radii'][:])
        self.masses = torch.from_numpy(file[group_name]['masses'][:])
        self.eps = torch.from_numpy(file[group_name]['eps'][:])
        self.n_cg_beads = int(file[group_name]['supdata'][0])
        self.binded_dna_len = int(file[group_name]['supdata'][2])
        return int(file[group_name]['supdata'][1])
        
    def get_true_pos(self,dna_structure=None,ref_om=None,ref_Rm=None):
        if ref_om is None:
            ref_om = torch.tensor(dna_structure.origins[self.ref_pair.ind]).to(self.ref_vectors.dtype)
            ref_Rm = torch.tensor(dna_structure.ref_frames[self.ref_pair.ind]).to(self.ref_vectors.dtype)

        return torch.matmul(self.ref_vectors.reshape(-1,1,3),ref_Rm.T) + ref_om
    
    def copy(self):
        return Protein(mdaUniverse=self.u,n_cg_beads=self.n_cg_beads,ref_pair = self.ref_pair,
                      eps=self.eps.clone(),ref_vectors = self.ref_vectors.clone(),
                      radii = self.radii.clone(),charges = self.charges.clone(),masses = self.masses.clone(),cg_structure=self.cg_structure,
                      binded_dna_len=self.binded_dna_len)
    
    def to(self,device):
        self.charges = self.charges.to(device)
        self.radii = self.radii.to(device)
        self.eps = self.eps.to(device)
        self.ref_vectors = self.ref_vectors.to(device)
    
    def _get_cg_centers(self,random_state):
        '''Runs optimization algorithm of CG beads positions according to procedure described in https://doi.org/10.1016/j.str.2006.10.003'''
        steps = 200*self.n_cg_beads
        
        non_H_atoms = self.u.select_atoms('not type H')
        max_coord = np.max(non_H_atoms.positions,axis=0)
        min_coord = np.min(non_H_atoms.positions,axis=0)
        origins = [np.random.uniform(min_v-10,max_v+10,(self.n_cg_beads,1)) 
                     for min_v,max_v in zip(min_coord,max_coord)]

        origins = np.hstack(origins)

        lmbd_0,eps_0 = 0.2*self.n_cg_beads,0.3
        lmbd_s,eps_s = 0.01,0.05

        atoms_number = len(non_H_atoms)
        for step in range(1,steps+1):
            rand = np.random.randint(0,atoms_number)
            ref_atom_r = non_H_atoms[rand].position
            dist = np.linalg.norm(ref_atom_r - origins,axis=1)
            k_beads = np.zeros(dist.shape)
            k_beads[np.argsort(dist)] = np.arange(self.n_cg_beads)

            lmbd = lmbd_0 * (lmbd_s/lmbd_0) ** (step/steps)
            eps = eps_0 * (eps_s/eps_0) ** (step/steps)

            origins = origins + eps * np.exp(-k_beads/lmbd).reshape(-1,1) * (ref_atom_r - origins)
        
        origins = torch.from_numpy(origins.reshape(-1,1,3))
        ref_om = self.ref_pair.om
        ref_Rm = self.ref_pair.Rm
        self.ref_vectors = torch.matmul((origins - ref_om),ref_Rm)
        

    
            
    def _get_cg_params(self):
        '''This method assigns each atom to a CG beads, than the charge of each bead is determined as a sum of charges of its atoms, and radii is defined by radius of gyration of atom groups of this CG bead.'''
        classifier = KNeighborsClassifier(1)
        classifier.fit(self.origins.reshape(-1,3),range(self.n_cg_beads))
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
        self.masses  = torch.from_numpy(self.masses)
        
    
    @property
    def origins(self):
        return self.get_true_pos(self.cg_structure.dna)