import torch

class _Real_Space_Energy_Calculator():
    def __init__(self,K_free,K_elec,include_elst,k_deb):
        
        self.K_free = K_free
        self.K_elec = K_elec
        self.k_deb = k_deb
        
        if include_elst:
            self.get_energy = self._get_real_space_total_energy
        else:
            self.get_energy = self._get_real_space_softmax_energy
            
        self.radii_sum_prod = None
        self.charges_multipl_prod = None
        self.dist_mat_slice = None
        self.es_en_mat = None
        self.sp_en_mat = None
        
    def to(self,device):
        radii_sum_prod,charges_multipl_prod,dist_mat_slice = self._energy_matrices()
        self.radii_sum_prod = radii_sum_prod.to(device)
        #self.epsilon_mean_prod = self.epsilon_mean_prod.to(device)
        self.charges_multipl_prod = charges_multipl_prod.to(device)
        self.dist_mat_slice = dist_mat_slice.to(device)
        if self.es_en_mat is not None:
            self.es_en_mat = self.es_en_mat.to(device)
        if self.sp_en_mat is not None:
            self.sp_en_mat = self.sp_en_mat.to(device)
            
        
    def set_energy_matrices(self,CG_structure,ignore_neighbors,ignore_protein_neigbors,set_dist_mat_sl):
        self.ignore_neighbors = ignore_neighbors
        self.ignore_protein_neigbors = ignore_protein_neigbors
        if set_dist_mat_sl:
            self._set_dist_mat_slice(CG_structure)
        radii = CG_structure.radii
        epsilons = CG_structure.eps
        charges = CG_structure.charges
        
        self.radii_sum_prod = (radii+radii.reshape(-1,1))
        self.charges_multipl_prod = torch.outer(charges,charges)
        self._mod_real_space_mat()

    def update_matrices(self,e_mat,s_mat,prot_ind):
        old_s_mat = self._get_matr_slices(self.sp_en_mat,prot_ind)
        old_e_mat = self._get_matr_slices(self.es_en_mat,prot_ind)
        # sl = old_s_mat == 0
        # old_s_mat[:] = s_mat
        # old_e_mat[:] = e_mat
        # old_s_mat[sl] = 0
        # old_e_mat[sl] = 0
        
    def get_energy_dif(self,prot_origins,prot_change_index):
        radii_sum_prod,charges_multipl_prod,_ = self._energy_matrices()
        
        radii_sum_prods = self._get_matr_slices(radii_sum_prod,prot_change_index)
        charges_multipl_prods = self._get_matr_slices(charges_multipl_prod,prot_change_index)
        
        dist_matrix = self._cdist(prot_origins[:prot_change_index],prot_origins[prot_change_index:])
        electrostatic2,e_mat = self._get_electrostatic_e(dist_matrix,charges_multipl_prods)
        spatial2,s_mat = self._get_spatial_e(dist_matrix,radii_sum_prods)
        
        old_s_mat = self._get_matr_slices(self.sp_en_mat,prot_change_index)
        old_e_mat = self._get_matr_slices(self.es_en_mat,prot_change_index)
        spatial1 = old_s_mat.sum()
        electrostatic1 = old_e_mat.sum()
        return electrostatic2-electrostatic1,spatial2-spatial1,e_mat,s_mat
            
    def _set_dist_mat_slice(self,CG_structure):
        prot_len = sum([protein.n_cg_beads for protein in CG_structure.proteins])
        dist_mat_slice = torch.ones(prot_len,prot_len,dtype=torch.bool)
        dist_mat_slice = torch.triu(dist_mat_slice)
        vert_start = 0 
        for protein in CG_structure.proteins:
            #This is incorrect, ignores by dna length, 2 accounts for protein size
            # dist_mat_slice[vert_start:vert_start+protein.binded_dna_len*2,
            #                vert_start:vert_start+protein.binded_dna_len*2] = False
            ignore_dist = max(self.ignore_protein_neigbors,protein.binded_dna_len//2)
            ignore_until = protein.ref_pair.ind + ignore_dist
            if ignore_until > prot_len:
                ignore_until = prot_len
            ignore_from = protein.ref_pair.ind - ignore_dist
            if ignore_from < 0:
                ignore_from = 0
            hor_start = 0
            for other_protein in CG_structure.proteins:
                if ignore_from <= other_protein.ref_pair.ind < ignore_until:
                    # print(vert_start,vert_start+protein.n_cg_beads,
                    #                 hor_start,hor_start+other_protein.n_cg_beads,ignore_dist)
                    dist_mat_slice[vert_start:vert_start+protein.n_cg_beads,
                                    hor_start:hor_start+other_protein.n_cg_beads] = False
                    dist_mat_slice[hor_start:hor_start+other_protein.n_cg_beads,
                                   vert_start:vert_start+protein.n_cg_beads] = False

                hor_start+=other_protein.n_cg_beads
                
            vert_start += protein.n_cg_beads
        
        self.dist_mat_slice = dist_mat_slice
        
    def _get_matr_slices(self,mat,prot_ind):

        return mat[:prot_ind,prot_ind:]

    def _energy_matrices(self):
        assert self.radii_sum_prod is not None
        assert self.charges_multipl_prod is not None
        assert self.dist_mat_slice is not None
        return self.radii_sum_prod,self.charges_multipl_prod,self.dist_mat_slice

    def _mod_real_space_mat(self):
        radii_sum_prod,charges_multipl_prod,dist_mat_slice = self._energy_matrices()
        inv = ~dist_mat_slice
        radii_sum_prod[inv] = 0
        charges_multipl_prod[inv] = 0
        
    
    def _get_real_space_total_energy(self,origins,prot_origins,save_matr=True):
        radii_sum_prod,charges_multipl_prod,dist_mat_slice = self._energy_matrices()

        total_es = torch.tensor(0,device=origins.device,dtype=origins.dtype)
        total_sp = torch.tensor(0,device=origins.device,dtype=origins.dtype)

        dist_matrix = self._cdist(prot_origins,prot_origins)
        dist_matrix = dist_matrix[dist_mat_slice]
        radii_sum_prod,charges_multipl_prod = radii_sum_prod[dist_mat_slice],charges_multipl_prod[dist_mat_slice]
        es,e_mat = self._get_electrostatic_e(dist_matrix,charges_multipl_prod)
        sp,s_mat = self._get_spatial_e(dist_matrix,radii_sum_prod)
        total_sp += sp
        total_es += es

        if save_matr:
            self.es_en_mat = torch.zeros(prot_origins.shape[0],prot_origins.shape[0],device=e_mat.device,dtype=e_mat.dtype)
            self.es_en_mat[dist_mat_slice] = e_mat
            self.sp_en_mat = torch.zeros(prot_origins.shape[0],prot_origins.shape[0],device=s_mat.device,dtype=s_mat.dtype)
            self.sp_en_mat[dist_mat_slice] = s_mat
                   
        return total_es,total_sp

    def _get_real_space_softmax_energy(self,origins,*args,**kwards):
        radii_sum_prod,_,dist_mat_slice = self._energy_matrices()

        dist_matrix = self._cdist(origins,origins)
        dist_matrix = dist_matrix[dist_mat_slice]
        radii_sum_prod = radii_sum_prod[dist_mat_slice]**2
        energy = self.K_free*(((radii_sum_prod/(dist_matrix**2+0.0001*radii_sum_prod))**6).sum())
        return torch.tensor(0,device=origins.device,dtype=energy.dtype),energy
    
    def _get_electrostatic_e(self,dist_matrix,charges_multipl_prod):
        div = charges_multipl_prod/dist_matrix
        exp = (self.k_deb*dist_matrix).exp()
        e_mat = div*exp*self.K_elec
        return e_mat.sum(),e_mat

    def _get_spatial_e(self,dist_matrix,radii_sum_prod):
        comp = (radii_sum_prod/dist_matrix).pow(6)
        s_mat = comp.pow(2).sub(comp)*self.K_free
        return s_mat.sum(),s_mat

    def _cdist(self,o1,o2):
        o1 = o1.reshape(-1,3)
        o2 = o2.reshape(-1,3)
        n = o1.size(0)
        m = o2.size(0)

        o1 = o1.unsqueeze(1).expand(n, m, 3)
        o2 = o2.unsqueeze(0).expand(n, m, 3)
        dist_mat = torch.pow(o2 - o1, 2).sum(2)
        
        return dist_mat.sqrt()