import torch

from pynamod.geometry.geometrical_parameters import Geometrical_Parameters

class All_Coords(Geometrical_Parameters):
    def __init__(self,local_params=None,origins=None,ref_frames=None,traj_len=None,proteins=None):
        self.prot_ind = {}
        super().__init__(local_params=local_params,origins=origins,ref_frames=ref_frames,traj_len=traj_len)
        if proteins:
            self.add_proteins(proteins)
        
    def add_proteins(self,proteins_list):
        start_index = self.len
        proteins_list = sorted(proteins_list,key = lambda x: - x.ref_pair.get_index())
        for protein in proteins_list:
            stop_index = start_index + protein.n_cg_beads
            self.prot_ind[protein.ref_pair.get_index()] = (start_index, stop_index,protein)
            start_index = stop_index
            
        prot_ori_frames = torch.zeros(self._origins_traj.shape[0],stop_index-self.len,1,3,dtype = self.dtype)
        self._origins_traj = torch.hstack([self._origins_traj,prot_ori_frames]) 
        
    def get_protein_origins(self,ref_index):
        start,end = self.prot_ind[ref_index][:2]
        return self.origins[start:end]
    
    def set_protein_origins(self,ref_index,value):
        start,end = self.prot_ind[ref_index][:2]
        self.origins[start:end] = value
        
    def rebuild_ref_frames_and_ori(self,start_index = 0, stop_index = None, start_ref_frame = None,start_origin = None,rebuild_proteins=False):
        super().rebuild_ref_frames_and_ori(start_index, stop_index, start_ref_frame,start_origin,rebuild_proteins)
        if rebuild_proteins:
            for ind_range in self.prot_ind.values():
                prot = ind_range[2] 
                self.origins[ind_range[0]:ind_range[1]] = prot.get_true_pos(prot.cg_structure.dna)
    
        
    def __getitem__(self,sl):
        it = Geometrical_Parameters.__getitem__(self,sl)
        sl = list(range(self.dna_len)[sl])
        for ref_ind,ind_range in self.prot_ind.items():
            ind_range = ind_range[:2]
            if ref_ind in sl:
                sl += list(range(*ind_range))
        
        it._origins_traj = self._origins_traj[:,sl]
        
        return it
    
    def __transform_ori(self,change_index,rot_matrix,prev_ori,changed_ori):
        stop = self.origins.shape[0]
        print(1)
        for ref_ind in sorted(self.prot_ind)[::-1]:
            print(1)
            if ref_ind < change_index:
                stop = self.prot_ind[ref_index][0]
                break
        
        self.origins[change_index+1:stop] -= prev_ori
        self.origins[change_index+1:stop] = self.origins[change_index+1:stop].matmul(rot_matrix.T) + changed_ori