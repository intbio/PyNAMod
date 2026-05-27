import torch

from pynamod.energy.elastic_energy import _Elastic_Energy_Calculator
from pynamod.energy.energy_constants import *
from pynamod.energy.external_forces.restraint import Restraint
from pynamod.energy.real_space_energy import _Real_Space_Energy_Calculator


class Energy:
    '''
    This class creates force matrices for a given CG structure and calculates its energy.
      '''
    def __init__(self,K_free=1,K_elec=1,K_bend=1,KT=300*8.314,salt_c=150,water_epsr = 81,include_elst=True):

        # Setting energy constants
        eps0 = 8.854
        eps0_order = -12
        dist_unit_order = -10
        q = 1.602
        q_order = -19
        na_order = 23
        na = 6.022
        kiloj_order = 3

        self._elastic_e_calc = _Elastic_Energy_Calculator(K_bend * KT * 10**(-kiloj_order))

        K_free = 0.001*KT/10**kiloj_order
        K_elec_order = q_order*2 - eps0_order - dist_unit_order
        K_elec = K_elec*(q**2)/(4*torch.pi*eps0*water_epsr*na)*10**(K_elec_order+na_order-kiloj_order)
        k_deb=-1/30
        #k_deb_order = (q_order*2+eps0_order)/2 + dist_unit_order 
        #k_deb = -(((2*salt_c*na*q**2)/(eps0*water_epsr*KT))**0.5)*10**k_deb_order
        self._real_space_calc = _Real_Space_Energy_Calculator(K_free,K_elec,include_elst,k_deb)

        self.KT = KT/10**kiloj_order
        self.restraints = []

    def set_energy_matrices(self,CG_structure,ignore_neighbors=5,set_dist_mat_sl=True):
        '''Creates matrices for energy calculation.

            Attributes:

            **CG_structure** - structure for which matrices are set.

            **ignore_neighbors** - number of neigboring dna pairs (in both sides) interactions with which are ignored in real space. Deafult 5.
            '''
        self._elastic_e_calc.set_energy_matrices(CG_structure)
        self._real_space_calc.set_energy_matrices(CG_structure,ignore_neighbors,set_dist_mat_sl)

    def add_restraints(self,restraints=None,restraint_type=None,CG_structure=None,scaling_factor=1):
        '''
        Attributes: 

        **restraints** - list of restraint objects.

        **restraint_type** - automatic generation of restraint. Could be 'circular_with_linear_restraint','circular_with_elastic_restraint'.
        '''
        if isinstance(restraints, list):
            self.restraints += restraints
        if restraint_type == 'circular_with_linear_restraint':
            self._get_circular_restraint('linear',CG_structure, scaling_factor)
        elif restraint_type == 'circular_with_elastic_restraint':
            self._get_circular_restraint('elastic',CG_structure, scaling_factor)

    def to(self,device):
        self._elastic_e_calc.to(device)
        self._real_space_calc.to(device)

        for restraint in self.restraints:
            restraint.to(device)

    def get_energy_components(self, params_storage, save_matr=True):
        elastic = self._elastic_e_calc.get_energy(params_storage.local_params)

        electrostatic, spatial = self._real_space_calc.get_energy(params_storage.origins,params_storage.rlsp_origins,save_matr=save_matr)
        restraint = self._get_restraint_energy(params_storage)
        return elastic, electrostatic, spatial, restraint

    def _get_restraint_energy(self, all_coords):
        if self.restraints:
            return sum([restraint.get_restraint_energy(all_coords) for restraint in self.restraints])
        else:
            return torch.tensor(0,dtype=torch.double,device=self._real_space_calc.radii_sum_prod[0].device)    

    def update_matrices(self, e_mat, s_mat, rlsp_change_index):
        self._real_space_calc.update_matrices(e_mat,s_mat,rlsp_change_index)

    def get_energy_dif(self, params_storage, rlsp_change_index, prev_e):

        del_electrostatic, del_spatial, e_mat, s_mat = self._real_space_calc.get_energy_dif(params_storage.rlsp_origins,rlsp_change_index)

        elastic1 = prev_e[0]
        restraint1 = prev_e[3]
        elastic2 = self._elastic_e_calc.get_energy(params_storage.local_params)
        restraint2 = self._get_restraint_energy(params_storage)
        return (elastic2-elastic1,del_electrostatic,del_spatial,restraint2-restraint1),e_mat,s_mat

    def _get_circular_restraint(self,restraint_func,CG_structure,scaling_factor):
        if restraint_func == 'elastic':
            #Might use incorrect pair
            pairtype = CG_structure.dna.pairs_list[0].pair_name[0] + CG_structure.dna.pairs_list[-1].pair_name[1]
            AVERAGE,FORCE_CONST,DISP = get_consts_olson_98()
            target = torch.tensor(AVERAGE[pairtype])
            const = torch.tensor(FORCE_CONST[pairtype])
        elif restraint_func == 'linear':
            target = torch.tensor(3.4)
            const = torch.tensor(0.4)
        dna_length = len(CG_structure.dna.pairs_list)
        self.restraints += [Restraint(0, dna_length-1, scaling_factor, target, const, en_restr_func=restraint_func)]
        self._real_space_calc.update_matrices_for_restraints(dna_length)
