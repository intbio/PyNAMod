import torch
import numpy as np
import h5py
from importlib.resources import files
from pynamod.structures.rlsp_group import Real_Space_Beads_Groups
from pynamod.atomic_analysis.nucleotides_parser import nucleotide_graphs, check_if_nucleotide


class Nucleotide_Creator:
    def __init__(self, model_type):
        if model_type == '1spn':
            self.create = self.create_1spn
        elif model_type == '3spn_from_atomic':
            self.create = self.create_3spn_from_atomic
            self.graphs_without_phosphate = {}
            for k,v in nucleotide_graphs.items():
                new = v.copy()
                new.remove_nodes_from([0,1,2])
                self.graphs_without_phosphate[k] = new

        elif model_type == '3spn_generated':
            self.create = self.create_3spn_generated
            self.ref_rlsps = {}
            for name in ['AT','TA','CG','GC']:
                rlspg = Real_Space_Beads_Groups(6)
                path = files('pynamod').joinpath(f'structures/ref_3spn/{name}.h5')
                rlspg.load_from_h5(h5py.File(path,'r'))
                self.ref_rlsps[name] = rlspg
                

        elif model_type == '1spbp':
            self.create = self.create_1spbp

    def create_1spn(self, pair, ref_ori, ref_r):
        lead_nucl_u = pair.lead_nucl.res_atoms
        lag_nucl_u = pair.lag_nucl.res_atoms

        radii = torch.tensor([lag_nucl_u.radius_of_gyration(), lead_nucl_u.radius_of_gyration()])
        charges = torch.tensor([-1, -1])
        origins = torch.from_numpy(np.vstack([lag_nucl_u.center_of_mass(),
                                              lead_nucl_u.center_of_mass()]).reshape(-1,1,3))
        ref_vectors = torch.matmul((origins - ref_ori), ref_r)

        return Real_Space_Beads_Groups(n_cg_beads=2, ref_pair=pair,
                                       ref_vectors=ref_vectors, charges=charges,
                                       radii=radii, binded_dna_len=1)

    def create_3spn_from_atomic(self, pair, ref_ori, ref_r, nucleotide_graphs=nucleotide_graphs):
        radii = []
        charges = []
        origins = []
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
            radii += nucl_radii
            charges += nucl_charges

        origins = torch.from_numpy(np.vstack(origins).reshape(-1, 1, 3))
        radii = torch.tensor(radii)
        charges = torch.tensor(charges)
        ref_vectors = torch.matmul((origins - ref_ori), ref_r)

        return Real_Space_Beads_Groups(n_cg_beads=radii.shape[0], ref_pair=pair,
                                       ref_vectors=ref_vectors, charges=charges,
                                       radii=radii, binded_dna_len=1)

    def create_3spn_generated(self, pair, ref_ori, ref_r):
        name = pair.lead_nucl.restype + pair.lag_nucl.restype
        rlspg = self.ref_rlsps[name].copy()
        rlspg.ref_pair = pair
        return rlspg

    def create_1spbp(self, pair, ref_ori, ref_r):
        radii = torch.tensor([10])
        charges = torch.tensor([-2])

        ref_vectors = torch.zeros(3)

        return Real_Space_Beads_Groups(n_cg_beads=1, ref_pair=pair,
                                       ref_vectors=ref_vectors, charges=charges,
                                       radii=radii, binded_dna_len=1)

