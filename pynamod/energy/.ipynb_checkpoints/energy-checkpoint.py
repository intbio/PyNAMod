import torch
from scipy.spatial.distance import squareform,cdist

from pynamod.energy.energy_constants import *
from pynamod.external_forces.restraint import Restraint


class Energy:
    '''This class creates force matrices for a given CG structure and calculates its energy.'''
    def __init__(self,K_free=1,K_elec=1,K_bend=1,KT=300*8.314,salt_c=150,water_epsr = 81,include_elst=True):
        
        eps0 = 8.854
        eps0_order = -12
        dist_unit_order = -10
        q = 1.602
        q_order = -19
        na_order = 23
        na = 6.022
        kiloj_order = 3
        self.force_matrix = None
        self.average_step_params = None
        self.K_free = K_free
        K_elec_order = q_order*2 - eps0_order - dist_unit_order
        self.K_elec = K_elec*(q**2)/(4*torch.pi*eps0*water_epsr*na)*10**(K_elec_order+na_order-kiloj_order)
        self.K_bend = K_bend * KT * 10**(-kiloj_order)
        #k_deb_order = (q_order*2+eps0_order)/2 + dist_unit_order 

        #self.k_deb = -(((2*salt_c*na*q**2)/(eps0*water_epsr*KT))**0.5)*10**k_deb_order
        self.eps = 0.001*KT/10**kiloj_order
        self.k_deb = -1/30
        self.KT = KT/10**kiloj_order
        self.restraints = []
        if include_elst:
            self.real_space_energy_func = self._get_real_space_total_energy
        else:
            self.real_space_energy_func = self._get_real_space_softmax_energy
        
    
    def set_energy_matrices(self,CG_structure,ignore_neighbors=5,ignore_protein_neigbors=8,set_dist_mat_sl=True):
        '''Creates matrices for energy calculation.
        
            Attributes:
            
            **CG_structure** - structure for which matrices are set.
            
            **ignore_neighbors** - number of neigboring dna pairs (in both sides) interactions with which are ignored in real space. Deafult 5.'''
        AVERAGE,FORCE_CONST,DISP = get_consts_olson_98()
        pairtypes = [pair.pair_name for pair in CG_structure.dna.pairs_list]
        self.ignore_neighbors = ignore_neighbors
        self.ignore_protein_neigbors = ignore_protein_neigbors
        if set_dist_mat_sl:
            self._set_dist_mat_slice(CG_structure)
        self._set_matrix(pairtypes,'force_matrix',FORCE_CONST)
        self._set_matrix(pairtypes,'average_step_params',AVERAGE)
        self._set_real_space_force_mat(CG_structure)
        
    
    def add_restraints(self,restraints=None,restraint_type=None,CG_structure=None,scaling_factor=1):
        '''
        Attributes: 
        
        **restraints** - list of restraint objects
        **restraint_type** - automatic generation of restraint. Could be 'circular_with_linear_restraint','circular_with_elastic_restraint'
        ''' 
        if isinstance(restraints,list):
            self.restraints += restraints
        if restraint_type == 'circular_with_linear_restraint':
            self._get_circular_restraint('linear',CG_structure,scaling_factor)
        elif restraint_type == 'circular_with_elastic_restraint':
            self._get_circular_restraint('elastic',CG_structure,scaling_factor)
            
    def _get_circular_restraint(self,restraint_func,CG_structure,scaling_factor):
        dna_length = CG_structure.dna.radii.shape[0]
        if restraint_func == 'elastic':
            pairtype = CG_structure.dna.pairs_list[0].pair_name[0] + CG_structure.dna.pairs_list[1].pair_name[1]
            AVERAGE,FORCE_CONST,DISP = get_consts_olson_98()
            target = torch.tensor(AVERAGE[pairtype])
            const = torch.tensor(FORCE_CONST[pairtype])
        elif restraint_func == 'linear':
            target = torch.tensor(3.4)
            const = torch.tensor(0.4)
        self.restraints += [Restraint(0,CG_structure.dna.origins.shape[0]-1,scaling_factor,target,const,en_restr_func=restraint_func)]
        self.dist_mat_slice[0:self.ignore_neighbors,
                            dna_length - self.ignore_neighbors:dna_length] = torch.tril(torch.ones(
                                            self.ignore_neighbors,self.ignore_neighbors,dtype=bool),diagonal=-1)   
        self.mod_real_space_mat()

            
    def to(self,device):
        self.force_matrix = self.force_matrix.to(device)
        self.average_step_params = self.average_step_params.to(device)
        self.radii_sum_prod = self.radii_sum_prod.to(device)
        #self.epsilon_mean_prod = self.epsilon_mean_prod.to(device)
        self.charges_multipl_prod = self.charges_multipl_prod.to(device)
        self.dist_mat_slice = self.dist_mat_slice.to(device)

        for restraint in self.restraints:
            restraint.to(device)
    
    def update_matrices(self,e_mat,s_mat,index,prot_sl_index):
        (sl1,sl2) = self._get_matr_slices(self.es_en_mat,index,prot_sl_index)
        sl1[:] = e_mat[:index]
        sl2[:] = e_mat[index:].T

        (sl1,sl2) = self._get_matr_slices(self.sp_en_mat,index,prot_sl_index)
        sl1[:] = s_mat[:index]
        sl2[:] = s_mat[index:].T
    
    def get_energy_components(self,params_storage,save_matr=True):
        elastic = self._get_elastic_energy(params_storage.local_params)
        if hasattr(params_storage,'prot_origins'):
            origins = torch.vstack([params_storage.origins,params_storage.prot_origins])
        else:
            origins = params_storage.origins
        electrostatic,spatial = self.real_space_energy_func(origins,save_matr=save_matr,square_mat_ln=origins.shape[0])
        restraint = self._get_restraint_energy(params_storage)
        return elastic,electrostatic,spatial,restraint
    
    def get_energy_dif(self,prev_e,static_origins,rotated_origins,cur_params_storage,change_index,prot_sl_index):
        
        radii_sum_prod = self._get_matr_slices(self.radii_sum_prod,change_index,prot_sl_index)
        radii_sum_prod = torch.vstack([radii_sum_prod[0],radii_sum_prod[1].T])
        charges_multipl_prod = self._get_matr_slices(self.charges_multipl_prod,change_index,prot_sl_index)
        charges_multipl_prod = torch.vstack([charges_multipl_prod[0],charges_multipl_prod[1].T])
            
        electrostatic1 = self._get_matr_slices(self.es_en_mat,change_index,prot_sl_index)
        electrostatic1 = (electrostatic1[0].sum() + electrostatic1[1].sum())*self.K_elec
        spatial1 = self._get_matr_slices(self.sp_en_mat,change_index,prot_sl_index)
        spatial1 = (spatial1[0].sum() + spatial1[1].sum())*self.K_free
        
        elastic1 = prev_e[0]
        restraint1 = prev_e[3]
        
        
        dist_matrix = torch.cdist(static_origins.reshape(-1,3),rotated_origins.reshape(-1,3))
        
        #return dist_matrix
        
        elastic2 = self._get_elastic_energy(cur_params_storage.local_params)
        electrostatic2,spatial2,e_mat,s_mat = self._get_real_space_slice_energy(dist_matrix,radii_sum_prod,charges_multipl_prod)
        restraint2 = self._get_restraint_energy(cur_params_storage)
        return (elastic2-elastic1,electrostatic2-electrostatic1,spatial2-spatial1,restraint2-restraint1),e_mat,s_mat
    
    
    def mod_real_space_mat(self):
        sl = torch.ones(*self.radii_sum_prod.shape,dtype=bool)
        sl[self.dist_mat_slice] = False
        self.radii_sum_prod[sl] = 0
        #self.epsilon_mean_prod[sl] = 0
        self.charges_multipl_prod[sl] = 0
        
     
    def _get_matr_slices(self,mat,ind,prot_ind):
        return (mat[:ind,ind:prot_ind],mat[ind:prot_ind,prot_ind:])
        
    def _set_matrix(self,pairtypes,attr,ref):
        matrix = torch.zeros((len(pairtypes),*ref['CG'].shape),dtype=torch.double)
        for i in range(len(pairtypes)-1):
            step = str(pairtypes[i][0]+pairtypes[i+1][0])
            matrix[i+1] = torch.Tensor(ref[step])
        setattr(self,attr,matrix)
    
    
    def _set_real_space_force_mat(self,CG_structure):
        radii = CG_structure.radii
        epsilons = CG_structure.eps
        charges = CG_structure.charges
        self.radii_sum_prod = (radii+radii.reshape(-1,1))
        #self.epsilon_mean_prod = torch.outer(epsilons,epsilons)/2
        self.charges_multipl_prod = torch.outer(charges,charges)
        
    
    def _get_real_space_triform(self):
        #self._triform(self.epsilon_mean_prod)
        return self._triform(self.radii_sum_prod),self._triform(self.charges_multipl_prod)
        
    
    def _set_dist_mat_slice(self,CG_structure):
        length = CG_structure.radii.shape[0]
        dna_length = CG_structure.dna.radii.shape[0]
        self.dist_mat_slice = torch.zeros(length,length,dtype=bool)
        dna_sl = torch.ones(dna_length,dna_length,dtype=bool)
        self.dist_mat_slice[:dna_length,:dna_length] = torch.triu(dna_sl, diagonal=self.ignore_neighbors)
        start = dna_length
        for protein in CG_structure.proteins[::-1]:
            stop = protein.n_cg_beads+start
            self.dist_mat_slice[0:start,start:stop] = True
            ref_ind = protein.ref_pair.ind
            offset = protein.binded_dna_len // 2 + self.ignore_protein_neigbors
            low = ref_ind-offset
            if low < 0:
                low = 0
            high = ref_ind + offset
            if high > self.dist_mat_slice.shape[0]:
                high = self.dist_mat_slice.shape[0]
            self.dist_mat_slice[low:high,start:stop] = False
            start = stop
        
    def _triform(self,square_mat):
        return square_mat[self.dist_mat_slice]
    

        
    def _get_elastic_energy(self,steps_params):
        params_dif = steps_params - self.average_step_params
        dif_matrix = torch.matmul(params_dif.reshape(-1,6,1), params_dif.reshape(-1,1,6))
        return self.K_bend*torch.einsum('ijk,ijk',dif_matrix, self.force_matrix)/2.0
    
    def _get_real_space_slice_energy(self,dist_matrix,radii_sum_prod,charges_multipl_prod):

        es,e_mat = self._get_electrostatic_e(dist_matrix,charges_multipl_prod)
        sp,s_mat = self._get_spatial_e(dist_matrix,radii_sum_prod)
        
        return es,sp,e_mat,s_mat
    
    def _get_real_space_total_energy(self,origins1,save_matr=True,square_mat_ln=None):
        dist_matrix = self._cdist(origins1,origins1)
        dist_matrix = self._triform(dist_matrix)
        
        
        radii_sum_prod,charges_multipl_prod = self._get_real_space_triform()
        es,e_mat = self._get_electrostatic_e(dist_matrix,charges_multipl_prod)
        sp,s_mat = self._get_spatial_e(dist_matrix,radii_sum_prod)
        if save_matr:
            self.es_en_mat = torch.zeros(square_mat_ln,square_mat_ln,dtype=e_mat.dtype,device=e_mat.device)
            self.es_en_mat[self.dist_mat_slice] = e_mat
            self.sp_en_mat = torch.zeros(square_mat_ln,square_mat_ln,dtype=s_mat.dtype,device=s_mat.device)
            self.sp_en_mat[self.dist_mat_slice] = s_mat
            
        return es,sp
    
    def _get_real_space_softmax_energy(self,origins,*args,**kwards):
        dist_matrix = self._cdist(origins,origins)
        dist_matrix = self._triform(dist_matrix)
        radii_sum_prod = self._triform(self.radii_sum_prod)**2
        energy = self.eps*(((radii_sum_prod/(dist_matrix**2+0.0001*radii_sum_prod))**6).sum())
        return torch.tensor(0,device = self.radii_sum_prod.device),energy
    
    def _get_electrostatic_e(self,dist_matrix,charges_multipl_prod):
        div = charges_multipl_prod/dist_matrix
        exp = (self.k_deb*dist_matrix).exp()
        e_mat = div*exp
        return e_mat.sum()*self.K_elec,e_mat

    def _get_spatial_e(self,dist_matrix,radii_sum_prod):
        comp = (radii_sum_prod/dist_matrix).pow(6)
        s_mat = comp.pow(2).sub(comp)*self.eps
        return s_mat.sum()*self.K_free,s_mat


    def _get_restraint_energy(self,all_coords):
        if self.restraints:
            return sum([restraint.get_restraint_energy(all_coords) for restraint in self.restraints])
        else:
            return torch.tensor(0,dtype=torch.double,device=self.radii_sum_prod.device)
        

    def _cdist(self,o1,o2):
        o1 = o1.reshape(-1,3)
        o2 = o2.reshape(-1,3)
        n = o1.size(0)
        m = o2.size(0)

        o1 = o1.unsqueeze(1).expand(n, m, 3)
        o2 = o2.unsqueeze(0).expand(n, m, 3)
        dist_mat = torch.pow(o2 - o1, 2).sum(2)
        
        return dist_mat.sqrt()
