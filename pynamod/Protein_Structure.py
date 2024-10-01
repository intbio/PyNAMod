import MDAnalysis as mda
import numpy as np
import io
import subprocess
import tempfile
import pypdb
from sklearn.neighbors import KNeighborsClassifier

class Protein:
    def __init__(self,mdaUniverse,n_cg_beads=50,ref_pair = None,eps=1):
        self.n_cg_beads = n_cg_beads
        
        pdb_temp = tempfile.NamedTemporaryFile(suffix='.pdb')
        pqr_temp = tempfile.NamedTemporaryFile(suffix='.pqr')
        mdaUniverse.atoms.write(pdb_temp.name)
        process = subprocess.run(['pdb2pqr', '--ff=AMBER','--log-level=CRITICAL', pdb_temp.name, pqr_temp.name])
        self.u = mda.Universe(pqr_temp.name)

        self.ref_vectors = np.zeros((n_cg_beads,3))
        self.ref_pair = ref_pair
        self.eps = np.tile(eps,n_cg_beads)
    
    def get_cg_centers(self):
        steps = 200*self.n_cg_beads
        
        non_H_atoms = self.u.select_atoms('not type H')
        max_coord = np.max(non_H_atoms.positions,axis=0)
        min_coord = np.min(non_H_atoms.positions,axis=0)
        self.cg_beads_pos = [np.random.uniform(min_v-10,max_v+10,(self.n_cg_beads,1)) 
                     for min_v,max_v in zip(min_coord,max_coord)]

        self.cg_beads_pos = np.hstack(self.cg_beads_pos)

        lmbd_0,eps_0 = 0.2*self.n_cg_beads,0.3
        lmbd_s,eps_s = 0.01,0.05

        atoms_number = len(non_H_atoms)
        for step in range(1,steps+1):
            rand = np.random.randint(0,atoms_number)
            ref_atom_r = non_H_atoms[rand].position
            dist = np.linalg.norm(ref_atom_r - self.cg_beads_pos,axis=1)
            k_beads = np.zeros(dist.shape)
            k_beads[np.argsort(dist)] = np.arange(self.n_cg_beads)

            lmbd = lmbd_0 * (lmbd_s/lmbd_0) ** (step/steps)
            eps = eps_0 * (eps_s/eps_0) ** (step/steps)

            self.cg_beads_pos = self.cg_beads_pos + eps * np.exp(-k_beads/lmbd).reshape(-1,1) * (ref_atom_r - self.cg_beads_pos)
        ref_om = self.ref_pair.om
        ref_Rm = self.ref_pair.Rm
        self.ref_vectors = np.matmul((self.cg_beads_pos - ref_om),ref_Rm)

            
    def get_cg_params(self):
        classifier = KNeighborsClassifier(1)
        classifier.fit(self.cg_beads_pos,range(self.n_cg_beads))
        groups = classifier.predict(self.u.atoms.positions)
        radii = np.zeros(self.n_cg_beads)
        self.cg_charges = np.zeros(self.n_cg_beads)
        self.cg_masses = np.zeros(self.n_cg_beads)
        for i in range(self.n_cg_beads):
            sel = self.u.atoms[groups==i]
            radii[i] = sel.radius_of_gyration()
            self.cg_charges[i] = np.sum(sel.charges)
            self.cg_masses[i] = np.sum(sel.masses)

        self.cg_radii = radii + 1
   
    
    def build_cg_model(self):
        self.get_cg_centers()
        self.get_cg_params()
        
    def get_true_pos(self):
        ref_om = self.ref_pair.om
        ref_Rm = self.ref_pair.Rm
        return np.matmul(self.ref_vectors,ref_Rm.T) + ref_om