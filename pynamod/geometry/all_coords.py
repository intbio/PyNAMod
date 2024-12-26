import torch

from pynamod.geometry.bp_step_geometry import Geometrical_Parameters

class All_Coords(Geometrical_Parameters):
    def __init__(self,cg_structure,traj_len):
        step_params = cg_structure.dna.step_params
        origins = cg_structure.origins
        ref_frames = cg_structure.dna.ref_frames
        self.dna_len = cg_structure.dna.origins.shape[0]
        self.prot_ind = {}
        for protein in cg_structure.proteins:
            self.prot_ind[protein.ref_pair.get_index()] = protein.n_cg_beads
        super().__init__(local_params=step_params,origins=origins,ref_frames=ref_frames,traj_len=traj_len)
        
    def __getitem__(self,sl):
        it = Geometrical_Parameters.__getitem__(self,sl)
        sl = list(range(self.dna_len)[sl])
        prot_sl = self.dna_len
        for ind,ln in self.prot_ind.items():
            if ind in sl:
                sl += list(range(prot_sl,prot_sl+ln))
            prot_sl += ln
        
        it._origins_traj = self._origins_traj[:,sl]
        
        return it