import torch
import numpy as np
import h5py
from importlib.resources import files
from pynamod.structures.rlsp_group import Real_Space_Beads_Groups
from pynamod.atomic_analysis.nucleotides_parser import nucleotide_graphs, check_if_nucleotide

class Nucleotide_Creator:
    def __init__(self, model_type):
        if model_type == '1spn_from_atomic':
            self.create = self._create_1spn_from_atomic

        elif model_type == '1spn_generated':
            self.create = self._load_model_for_generated
            self.ref_rlsps = {}
            for name in ['AT', 'TA', 'CG', 'GC']:
                rlspg = Real_Space_Beads_Groups(2)
                path = files('pynamod').joinpath(f'structures/ref_cg_pairs/1spn/{name}.h5')
                rlspg.load_from_h5(h5py.File(path, 'r')['rlspg_0_CG_parameters'])
                self.ref_rlsps[name] = rlspg

        elif model_type == '3spn_from_atomic':
            self.create = self._create_3spn_from_atomic
            self.graphs_without_phosphate = {}
            for k,v in nucleotide_graphs.items():
                new = v.copy()
                new.remove_nodes_from([0, 1, 2])
                self.graphs_without_phosphate[k] = new
            self.masses_dict = {
                'P': 94.97,
                'S': 83.11,
                'Ab': 134.1,
                'Tb': 125.1,
                'Cb': 110.1,
                'Gb': 150.1
            }

        elif model_type == '3spn_generated':
            self.create = self._load_model_for_generated
            self.ref_rlsps = {}
            for name in ['AT', 'TA', 'CG', 'GC']:
                rlspg = Real_Space_Beads_Groups(6)
                path = files('pynamod').joinpath(f'structures/ref_cg_pairs/3spn/{name}.h5')
                rlspg.load_from_h5(h5py.File(path, 'r')['rlspg_0_CG_parameters'])
                self.ref_rlsps[name] = rlspg

        elif model_type == '1spbp_from_atomic' or model_type == '1spbp_generated':
            self.create = self._create_1spbp
            self.masses_dict = {
                'AT': (94.97 + 83.11)*2 + 134.1 + 125.1,
                'CG': (94.97 + 83.11)*2 + 110.1 + 150.1
            }

        else:
            raise AttributeError(f"Unknown nucleotide model '{model_type}'")

    def _create_1spn_from_atomic(self, pair, ref_ori, ref_r):
        lead_nucl_u = pair.lead_nucl.res_atoms
        lag_nucl_u = pair.lag_nucl.res_atoms

        radii = torch.tensor([lag_nucl_u.radius_of_gyration(), lead_nucl_u.radius_of_gyration()])
        charges = torch.tensor([-1, -1])
        origins = torch.from_numpy(np.vstack([lag_nucl_u.center_of_mass(),
                                              lead_nucl_u.center_of_mass()]).reshape(-1,1,3))
        masses = torch.tensor([lag_nucl_u.masses.sum(), lead_nucl_u.masses.sum()])
        ref_vectors = torch.matmul((origins - ref_ori), ref_r)

        return self._init_rlspg(pair.ind, radii.shape[0], ref_vectors,
                                charges, radii, masses)

    def _create_3spn_from_atomic(self, pair, ref_ori, ref_r):
        radii = []
        charges = []
        origins = []
        masses = []
        for nucl in (pair.lag_nucl, pair.lead_nucl):
            nucl_u = check_if_nucleotide(nucl.res_atoms, use_full_nucleotide=True)
            nucl_radii = [3.4]*3
            nucl_charges = [-1, 0, 0]
            if nucl_u[2] == '':
                nucl_u = check_if_nucleotide(nucl.res_atoms, nucleotide_graphs=self.graphs_without_phosphate, use_full_nucleotide=True)
                phosphate_removed = True
                nucl_radii = nucl_radii[1:]
                nucl_charges = nucl_charges[1:]
            else:
                phosphate_removed = False

            nucl_u = sum(nucl_u[0])
            if not phosphate_removed:
                phosphate_origin = nucl_u[0:3].center_of_mass().astype(np.float64)
                origins += [phosphate_origin]
                masses += [self.masses_dict['P']]
            sugar_origin = nucl_u[3:11].center_of_mass().astype(np.float64)
            # In 3spn base bead position is defined by N3 atom position for purines and N1 position for pyrimidines
            if phosphate_removed:
                pyr_n1_index = 11
                pur_n3_index = 14
            else:
                pyr_n1_index = 14
                pur_n3_index = 17
            if nucl.restype in ('T', 'C'):
                base_origin = nucl_u[pyr_n1_index].position.astype(np.float64)
            elif nucl.restype in ('A', 'G'):
                base_origin = nucl_u[pur_n3_index].position.astype(np.float64)

            origins += [sugar_origin, base_origin]
            masses += [self.masses_dict['S'], self.masses_dict[f'{nucl.restype}b']]
            radii += nucl_radii
            charges += nucl_charges

        origins = torch.from_numpy(np.vstack(origins).reshape(-1, 1, 3))
        masses = torch.tensor(masses)
        radii = torch.tensor(radii)
        charges = torch.tensor(charges)
        ref_vectors = torch.matmul((origins - ref_ori), ref_r)

        return self._init_rlspg(pair.ind, radii.shape[0], ref_vectors,
                                charges, radii, masses)

    def _load_model_for_generated(self, pair, ref_ori, ref_r):
        name = pair.lead_nucl.restype + pair.lag_nucl.restype
        rlspg = self.ref_rlsps[name].copy()
        rlspg.ref_ind = pair.ind
        return rlspg

    def _create_1spbp(self, pair, ref_ori, ref_r):
        radii = torch.tensor([10])
        charges = torch.tensor([-2])
        masses = torch.tensor([self.masses_dict[pair.pair_name]])
        ref_vectors = torch.zeros(1, 1, 3)

        return self._init_rlspg(pair.ind, radii.shape[0], ref_vectors,
                                charges, radii, masses)

    def _init_rlspg(self, ind, n_cg_beads, ref_vectors, charges, radii, masses):
        return Real_Space_Beads_Groups(n_cg_beads=n_cg_beads, ref_ind=ind,
                                       ref_vectors=ref_vectors, charges=charges,
                                       masses=masses, radii=radii, binded_dna_len=1,
                                       group_type='DNA Pair')