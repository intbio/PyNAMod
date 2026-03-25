import torch
from scipy.spatial.distance import cdist, squareform

from pynamod.energy.energy_constants import *
from pynamod.external_forces.restraint import Restraint


class Energy:
    '''
    This class creates force matrices for a given CG structure and calculates its energy.
      '''

    def __init__(self, K_free=1, K_elec=1, K_bend=1, KT=300*8.314, salt_c=150, water_epsr=81, include_elst=True):

        # Setting energy constants
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
        # k_deb_order = (q_order*2+eps0_order)/2 + dist_unit_order

        # self.k_deb = -(((2*salt_c*na*q**2)/(eps0*water_epsr*KT))**0.5)*10**k_deb_order
        self.eps = 0.001*KT/10**kiloj_order
        self.k_deb = -1/30
        self.KT = KT/10**kiloj_order
        self.restraints = []
        if include_elst:
            self.real_space_energy_func = self._get_real_space_total_energy
        else:
            self.real_space_energy_func = self._get_real_space_softmax_energy

    def set_energy_matrices(self, CG_structure, ignore_neighbors=5, ignore_protein_neigbors=8, set_dist_mat_sl=True):
        '''Creates matrices for energy calculation.

            Attributes:

            **CG_structure** - structure for which matrices are set.

            **ignore_neighbors** - number of neigboring dna pairs (in both sides) interactions with which are ignored in real space. Deafult 5.
            '''
        AVERAGE, FORCE_CONST, DISP = get_consts_olson_98()
        pairtypes = [pair.pair_name for pair in CG_structure.dna.pairs_list]
        self._set_matrix(pairtypes, 'force_matrix', FORCE_CONST)
        self._set_matrix(pairtypes, 'average_step_params', AVERAGE)

        # real space matrices
        self.ignore_neighbors = ignore_neighbors
        self.ignore_protein_neigbors = ignore_protein_neigbors
        if set_dist_mat_sl:
            self._set_dist_mat_slice(CG_structure)
        radii = CG_structure.radii
        epsilons = CG_structure.eps
        charges = CG_structure.charges

        dna_len = len(CG_structure.dna.pairs_list)
        self.radii_sum_prod = [(radii[:dna_len]+radii[:dna_len].reshape(-1, 1)),
                               (radii[dna_len:]+radii[:dna_len].reshape(-1, 1)),
                               (radii[dna_len:]+radii[dna_len:].reshape(-1, 1))]

        self.charges_multipl_prod = [torch.outer(charges[:dna_len], charges[:dna_len]),
                                     torch.outer(charges[:dna_len], charges[dna_len:]),
                                     torch.outer(charges[dna_len:], charges[dna_len:])]

        self._mod_real_space_mat()

    def add_restraints(self, restraints=None, restraint_type=None, CG_structure=None, scaling_factor=1):
        '''
        Attributes:

        **restraints** - list of restraint objects.

        **restraint_type** - automatic generation of restraint. Could be 'circular_with_linear_restraint','circular_with_elastic_restraint'.
        '''
        if isinstance(restraints, list):
            self.restraints += restraints
        if restraint_type == 'circular_with_linear_restraint':
            self._get_circular_restraint('linear', CG_structure, scaling_factor)
        elif restraint_type == 'circular_with_elastic_restraint':
            self._get_circular_restraint('elastic', CG_structure, scaling_factor)

    def to(self, device):
        self.force_matrix = self.force_matrix.to(device)
        self.average_step_params = self.average_step_params.to(device)
        self.radii_sum_prod = [m.to(device) for m in self.radii_sum_prod]
        # self.epsilon_mean_prod = self.epsilon_mean_prod.to(device)
        self.charges_multipl_prod = [m.to(device) for m in self.charges_multipl_prod]
        self.dist_mat_slice = [m.to(device) for m in self.dist_mat_slice]

        for restraint in self.restraints:
            restraint.to(device)

    def update_matrices(self, e_mat, s_mat, change_indices):
        old_s_mat = self._get_matr_slices(self.sp_en_mat, *change_indices)
        old_e_mat = self._get_matr_slices(self.es_en_mat, *change_indices)

        for i in range(4):

            old_s_mat[i][:] = s_mat[i]
            old_e_mat[i][:] = e_mat[i]

    def get_energy_components(self, params_storage, save_matr=True):
        elastic = self._get_elastic_energy(params_storage.local_params)

        electrostatic, spatial = self.real_space_energy_func(params_storage.origins, params_storage.prot_origins, save_matr=save_matr)
        restraint = self._get_restraint_energy(params_storage)
        return elastic, electrostatic, spatial, restraint

    def get_energy_dif(self, params_storage, change_indices, prev_e):
        dna_change_index, prot_change_index = change_indices
        electrostatic2 = torch.tensor(0, device=params_storage.origins.device, dtype=params_storage.origins.dtype)
        spatial2 = torch.tensor(0, device=params_storage.origins.device, dtype=params_storage.origins.dtype)

        all_s_mat = []
        all_e_mat = []

        radii_sum_prods = self._get_matr_slices(self.radii_sum_prod, *change_indices)
        charges_multipl_prods = self._get_matr_slices(self.charges_multipl_prod, *change_indices)
        mat_ind = 0

        for i, static_origins in enumerate((params_storage.origins[:dna_change_index], params_storage.prot_origins[:prot_change_index])):
            for j, changed_origins in enumerate((params_storage.origins[dna_change_index:], params_storage.prot_origins[prot_change_index:])):
                change_ind1 = change_indices[i]
                change_ind2 = change_indices[j]
                if i == 1 and j == 0:
                    dist_matrix = self._cdist(changed_origins, static_origins)
                else:
                    dist_matrix = self._cdist(static_origins, changed_origins)
                electrostatic, e_mat = self._get_electrostatic_e(dist_matrix, charges_multipl_prods[mat_ind])
                spatial, s_mat = self._get_spatial_e(dist_matrix, radii_sum_prods[mat_ind])

                mat_ind += 1
                electrostatic2 += electrostatic
                spatial2 += spatial

                all_e_mat.append(e_mat)
                all_s_mat.append(s_mat)

        elastic1 = prev_e[0]
        restraint1 = prev_e[3]
        old_s_mat = self._get_matr_slices(self.sp_en_mat, *change_indices)
        old_e_mat = self._get_matr_slices(self.es_en_mat, *change_indices)
        spatial1 = sum([mat.sum() for mat in old_s_mat])
        electrostatic1 = sum([mat.sum() for mat in old_e_mat])
        elastic2 = self._get_elastic_energy(params_storage.local_params)
        restraint2 = self._get_restraint_energy(params_storage)
        return (elastic2-elastic1, electrostatic2-electrostatic1, spatial2-spatial1, restraint2-restraint1), all_e_mat, all_s_mat

    def _mod_real_space_mat(self):
        for i in range(3):
            inv = ~self.dist_mat_slice[i]
            self.radii_sum_prod[i][inv] = 0
            self.charges_multipl_prod[i][inv] = 0

    def _get_circular_restraint(self, restraint_func, CG_structure, scaling_factor):
        dna_length = CG_structure.dna.radii.shape[0]
        if restraint_func == 'elastic':
            # Might use incorrect pair
            pairtype = CG_structure.dna.pairs_list[0].pair_name[0] + CG_structure.dna.pairs_list[-1].pair_name[1]
            AVERAGE, FORCE_CONST, DISP = get_consts_olson_98()
            target = torch.tensor(AVERAGE[pairtype])
            const = torch.tensor(FORCE_CONST[pairtype])
        elif restraint_func == 'linear':
            target = torch.tensor(3.4)
            const = torch.tensor(0.4)
        self.restraints += [Restraint(0, CG_structure.dna.origins.shape[0]-1, scaling_factor, target, const, en_restr_func=restraint_func)]
        self.dist_mat_slice[0][0:self.ignore_neighbors,
                               dna_length - self.ignore_neighbors:dna_length] = torch.tril(torch.ones(
                                   self.ignore_neighbors, self.ignore_neighbors, dtype=bool), diagonal=-1)
        self._mod_real_space_mat()

    def _get_matr_slices(self, mat, dna_ind, prot_ind):
        return (mat[0][:dna_ind, dna_ind:], mat[1][:dna_ind, prot_ind:],
                mat[1][dna_ind:, :prot_ind], mat[2][:prot_ind, prot_ind:])

    def _set_matrix(self, pairtypes, attr, ref):
        matrix = torch.zeros((len(pairtypes), *ref['CG'].shape), dtype=torch.double)
        for i in range(len(pairtypes)-1):
            step = str(pairtypes[i][0]+pairtypes[i+1][0])
            matrix[i+1] = torch.tensor(ref[step])
        setattr(self, attr, matrix)

    def _set_dist_mat_slice(self, CG_structure):
        prot_len = sum([protein.n_cg_beads for protein in CG_structure.proteins])
        dna_len = len(CG_structure.dna.pairs_list)

        dna_dna_sl = torch.ones(dna_len, dna_len, dtype=bool)
        dna_dna_sl = torch.triu(dna_dna_sl, diagonal=self.ignore_neighbors+1)

        dna_prot_slice = torch.ones(dna_len, prot_len, dtype=bool)

        prot_prot_slice = torch.ones(prot_len, prot_len, dtype=bool)
        prot_prot_slice = torch.triu(prot_prot_slice)
        start = 0
        for protein in CG_structure.proteins:
            n_beads = protein.n_cg_beads
            prot_prot_slice[start:start+n_beads, start:start+n_beads] = False

            ref_ind = protein.ref_pair.ind
            dna_offset = protein.binded_dna_len // 2 + self.ignore_protein_neigbors
            low = ref_ind-dna_offset
            if low < 0:
                low = 0
            high = ref_ind + dna_offset
            if high > dna_prot_slice.shape[0]:
                high = dna_prot_slice.shape[0]

            dna_prot_slice[low:high, start:start+n_beads] = False

            start += n_beads

        self.dist_mat_slice = [dna_dna_sl, dna_prot_slice, prot_prot_slice]

    def _get_elastic_energy(self, steps_params):
        params_dif = steps_params - self.average_step_params
        dif_matrix = torch.matmul(params_dif.reshape(-1, 6, 1), params_dif.reshape(-1, 1, 6))
        return self.K_bend*torch.einsum('ijk,ijk', dif_matrix, self.force_matrix)/2.0

    def _get_real_space_total_energy(self, origins, prot_origins, save_matr=True):
        origins_comb = [(origins, origins),
                        (origins, prot_origins),
                        (prot_origins, prot_origins)
                        ]

        if save_matr:
            self.es_en_mat = []
            self.sp_en_mat = []
        total_es = torch.tensor(0, device=origins.device, dtype=origins.dtype)
        total_sp = torch.tensor(0, device=origins.device, dtype=origins.dtype)
        for i, ori_set in enumerate(origins_comb):
            dist_matrix = self._cdist(*ori_set)
            dist_matrix = dist_matrix[self.dist_mat_slice[i]]
            radii_sum_prod, charges_multipl_prod = self.radii_sum_prod[i][self.dist_mat_slice[i]], self.charges_multipl_prod[i][self.dist_mat_slice[i]]
            es, e_mat = self._get_electrostatic_e(dist_matrix, charges_multipl_prod)
            sp, s_mat = self._get_spatial_e(dist_matrix, radii_sum_prod)
            total_sp += sp
            total_es += es

            if save_matr:
                full_e_mat = torch.zeros(ori_set[0].shape[0], ori_set[1].shape[0], device=e_mat.device, dtype=e_mat.dtype)
                full_e_mat[self.dist_mat_slice[i]] = e_mat
                full_s_mat = torch.zeros(ori_set[0].shape[0], ori_set[1].shape[0], device=s_mat.device, dtype=s_mat.dtype)
                full_s_mat[self.dist_mat_slice[i]] = s_mat

                self.es_en_mat.append(full_e_mat)
                self.sp_en_mat.append(full_s_mat)
        return total_es, total_sp

    def _get_real_space_softmax_energy(self, origins, *args, **kwards):
        dist_matrix = self._cdist(origins, origins)
        dist_matrix = dist_matrix[self.dist_mat_slice]
        radii_sum_prod = self.radii_sum_prod[self.dist_mat_slice]**2
        energy = self.eps*(((radii_sum_prod/(dist_matrix**2+0.0001*radii_sum_prod))**6).sum())
        return torch.tensor(0, device=self.radii_sum_prod.device), energy

    def _get_electrostatic_e(self, dist_matrix, charges_multipl_prod):
        div = charges_multipl_prod/dist_matrix
        exp = (self.k_deb*dist_matrix).exp()
        e_mat = div*exp*self.K_elec
        return e_mat.sum(), e_mat

    def _get_spatial_e(self, dist_matrix, radii_sum_prod):
        comp = (radii_sum_prod/dist_matrix).pow(6)
        s_mat = comp.pow(2).sub(comp)*self.eps*self.K_free
        return s_mat.sum(), s_mat

    def _get_restraint_energy(self, all_coords):
        if self.restraints:
            return sum([restraint.get_restraint_energy(all_coords) for restraint in self.restraints])
        else:
            return torch.tensor(0, dtype=torch.double, device=self.radii_sum_prod[0].device)

    def _cdist(self, o1, o2):
        o1 = o1.reshape(-1, 3)
        o2 = o2.reshape(-1, 3)
        n = o1.size(0)
        m = o2.size(0)

        o1 = o1.unsqueeze(1).expand(n, m, 3)
        o2 = o2.unsqueeze(0).expand(n, m, 3)
        dist_mat = torch.pow(o2 - o1, 2).sum(2)

        return dist_mat.sqrt()
