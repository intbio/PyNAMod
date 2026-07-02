import torch
from pynamod.energy.energy_constants import *

class _Elastic_Energy_Calculator:
    def __init__(self,K_bend,multimodal=True):
        self.K_bend = K_bend
        self.unimodal_force_matrix = None
        self.unimodal_average_step_params = None
        self.multimodal = multimodal
        if multimodal:
            self._energy_func = self._get_multimodal_energy
        else:
            self._energy_func = self._get_unimodal_energy

    def set_energy_matrices(self,CG_structure):
        '''Creates matrices for energy calculation.

            Attributes:

            **CG_structure** - structure for which matrices are set.

            **ignore_neighbors** - number of neigboring dna pairs (in both sides) interactions with which are ignored in real space. Deafult 5.
            '''
        AVERAGE,FORCE_CONST,DISP = get_consts_olson_98()
        if not self.multimodal:
            step_types = [str(pair0.lead_nucl.restype + pair1.lead_nucl.restype)
                             for pair0,pair1 in
                             zip(CG_structure.dna.pairs[:-1],CG_structure.dna.pairs[1:])]
        else:
            step_types = [str(CG_structure.dna.pairs[i].lead_nucl.restype +
                               CG_structure.dna.pairs[i+1].lead_nucl.restype)
                          for i in (0,-2)
                          ]
        self._set_unimodal_matrix(step_types,'unimodal_force_matrix',FORCE_CONST)
        self._set_unimodal_matrix(step_types,'unimodal_average_step_params',AVERAGE)
        if self.multimodal:
            self._set_multimodal_params(CG_structure)

    def _set_multimodal_params(self,CG_structure):
        tetra_seqs = [
            "".join(
                CG_structure.dna.pairs[j].lead_nucl.restype
                for j in range(i - 2, i + 2)
            )
            for i in range(2, CG_structure.dna.step_params.shape[0] - 1)
        ]

        max_modes = max(
            len(tetra_params_dict[s]["e_state"])
            for s in tetra_seqs
        )

        self._set_multimodal_matrix(tetra_seqs,max_modes,'e_state',tetra_params_dict)
        self._set_multimodal_matrix(tetra_seqs,max_modes,'force_matrix',tetra_params_dict)
        self._set_multimodal_matrix(tetra_seqs,max_modes,'average_params',tetra_params_dict)
        self.multi_mask = torch.zeros(len(CG_structure.dna.pairs)-3, max_modes,dtype=torch.bool)
        for i,seq in enumerate(tetra_seqs):
            n_modes = tetra_params_dict[seq]['e_state'].shape[0]
            self.multi_mask[i, :n_modes] = True


    def to(self,device):
        self.unimodal_force_matrix = self.unimodal_force_matrix.to(device)
        self.unimodal_average_step_params = self.unimodal_average_step_params.to(device)

    def get_energy(self,step_params):
        return self._energy_func(step_params)

    def _set_unimodal_matrix(self,step_types,attr,ref):
        matrix = torch.zeros((len(step_types),*ref['CG'].shape),dtype=torch.double)

        for i,step_type in enumerate(step_types):
            matrix[i] = torch.tensor(ref[step_type])
        setattr(self,attr,matrix)

    def _set_multimodal_matrix(self,pairtypes,max_modes,attr,tetra_dict_params):
        matrix = torch.zeros(
            (len(pairtypes), max_modes, *tetra_dict_params['ATCG'][attr].shape[1:]),
            dtype=torch.double
        )

        for i, seq in enumerate(pairtypes):
            step_matrix = torch.tensor(tetra_dict_params[seq][attr])
            matrix[i,:step_matrix.shape[0]] = step_matrix
        setattr(self,'multimodal_' + attr,matrix)

    def _get_unimodal_energy(self,step_params):
        params_dif = step_params[1:] - self.unimodal_average_step_params
        dif_matrix = torch.matmul(params_dif.reshape(-1,6,1), params_dif.reshape(-1,1,6))
        return self.K_bend*torch.einsum('ijk,ijk',dif_matrix, self.unimodal_force_matrix)/2.0

    def _get_multimodal_energy(self,step_params):
        print(step_params[1,-1])
        uni_e = self._get_unimodal_energy(step_params[[1,-1]])
        if step_params.shape[0] > 2:
            step_params = step_params[2:-1]
            dx = step_params[:, None, :] - self.multimodal_average_params

            quadratic = 0.5 * torch.einsum(
                "bmi,bmij,bmj->bm",
                dx,
                self.multimodal_force_matrix,
                dx,
            )

            powers = quadratic + self.multimodal_e_state

            powers = powers.masked_fill(
                ~self.multi_mask,
                torch.inf,
            )

            total_e = (-self.K_bend * torch.logsumexp(
                -powers / self.K_bend,
                dim=1
            )).sum()
        print(uni_e,total_e)
        return uni_e,total_e
