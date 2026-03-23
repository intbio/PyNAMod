import torch
from sklearn.neighbors import KNeighborsClassifier
from itertools import combinations

class SAXS_Fit:
    def __init__(self,cgs,target_pdd,force_const,dist_mat_slice,cdist_f):
        self.target_pdd = torch.tensor(target_pdd)
        self.var = force_const
        self.set_constants()
        self.get_total_dz_list(cgs)
        self.dist_mat_slice = torch.triu(torch.ones(*dist_mat_slice.shape,dtype=bool),diagonal=1)
        self.cdist_f = cdist_f
        self.force_const = force_const
        
    def set_constants(self):
        r_dict = {'C':170,'H':110,'N':155,'O':152,'P':180,'S':180}
        self.n_e = {'C':6,'H':1,'N':7,'O':8,'P':15,'S':16}
        self.volumes = {t:4/3*torch.pi*(r_dict[t]*10**-2)**3 for t in r_dict.keys()}
        Na = 6.02214076*10**23
        self.ED = (1000/18* 10 + 0.15*28)*Na / 0.001 / 10**30
        
    def get_total_dz_list(self,cgs):
        dz_list = []
        ref_ind_list = []
        for pair in cgs.dna.pairs_list:
            for n in (pair.lead_nucl,pair.lag_nucl):
                nucl_u = cgs.u.select_atoms(f'resid {n.resid} and segid {n.segid}')
                dz = self.get_coarse_bead_dZ(nucl_u)
                dz_list.append(dz)
                ref_ind_list.append(pair.ind)
        proteins_dz = []
        for protein in cgs.proteins:
            if protein.n_cg_beads > 2:
                proteins_dz += self.get_protein_dZ_list(protein)
                ref_ind_list += [protein.ref_pair.ind]*protein.n_cg_beads
        cgs_dz_list = dz_list + proteins_dz
        sort_ind = sorted(range(len(ref_ind_list)), key=lambda i: ref_ind_list[i])

        cgs_dz_list_sorted = [cgs_dz_list[i] for i in sort_ind]
        self.weights = torch.tensor([dz1*dz2 for dz1,dz2 in combinations(cgs_dz_list_sorted,2)]).to(torch.float)
        
    def get_dZ(self,atom_type):
        return self.n_e[atom_type] - self.volumes[atom_type]*self.ED
        
    def get_coarse_bead_dZ(self,u):
        dz = 0
        for atom in u.atoms:
            dz += self.get_dZ(atom.type)

        return dz
    
    def get_protein_dZ_list(self,cg_protein):
        classifier = KNeighborsClassifier(1)
        classifier.fit(cg_protein.origins.reshape(-1,3),range(cg_protein.n_cg_beads))
        protein_u = cg_protein.u.select_atoms('not type H')
        groups = classifier.predict(protein_u.positions)
        dz_list = []
        for i in range(cg_protein.n_cg_beads):
            atoms = protein_u[groups==i]
            dz_list.append(self.get_coarse_bead_dZ(atoms))

        return dz_list
    
    def to(self,device):
        pass
        # self.target_pdd = torch.tensor(self.target_pdd).to(device)
        # self.weights = self.weights.to(device)
        
    def get_hist(self,prot_origins):
        pd = self.cdist_f(prot_origins,prot_origins)[self.dist_mat_slice]
        
        pdd_exp,r = torch.histogram(pd.cpu(),bins=self.target_pdd.shape[0],weight=self.weights)
        pdd_exp = pdd_exp/pdd_exp.sum()
        return pdd_exp,r
        
    def get_restraint_energy(self,intg_trajectory):
        pd = self.cdist_f(intg_trajectory.prot_origins,intg_trajectory.prot_origins)[self.dist_mat_slice]
        
        pdd_exp,r = torch.histogram(pd.cpu(),bins=self.target_pdd.shape[0],weight=self.weights)
        pdd_exp = pdd_exp/pdd_exp.sum()
        #return pdd_exp,r
        return (((self.target_pdd - pdd_exp)**2).sum()*self.force_const).to('cuda')
        
        

        
    