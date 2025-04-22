import torch

class mod_Tensor(torch.Tensor):
    geom_class = None
    def __new__(cls, x,geom_class, *args, **kwargs):
        return super().__new__(cls, x, *args, **kwargs)
    
    def __init__(self,x,geom_class,*args,**kwards):
        self.geom_class = geom_class


    def __getitem__(self, sl):
        it = super().__getitem__(sl)
        it.geom_class = self.geom_class
        return it

    def __setitem__(self, sl, value):

        super().__setitem__(sl,value)
                           
        
        if self.geom_class and self.geom_class._auto_rebuild_sw:
            is_traj = (self.dim() == 3 and self.shape[-1] == 6) or self.dim() > 3
            full_frame_changed = (isinstance(sl,int) or isinstance(sl,torch.Tensor)) and is_traj
            if isinstance(sl,tuple):
                sl = sl[0]
            
            if isinstance(sl,slice) or full_frame_changed:
                if self.shape[-1] == 3 or self.shape[-1] == 4:
                    self.geom_class.rebuild('rebuild_local_params')
                elif self.shape[-1] == 6:
                    self.geom_class.rebuild('rebuild_ref_frames_and_ori',rebuild_proteins=True)

            elif isinstance(sl,int) or (isinstance(sl,torch.Tensor) and sl.shape == tuple()):
                if self.shape[-1] == 3 or self.shape[-1] == 4:
                    self.geom_class.rebuild('rebuild_local_params',start_index=sl)
                elif self.shape[-1] == 6:
                    self.geom_class.rebuild('rotate_ref_frames_and_ori',sl)

                    
                    
class Origins_Tensor(mod_Tensor):
    geom_class = None
    def __new__(cls, x,geom_class,protein_data=None,proteins_list=None, *args, **kwargs):
        return super().__new__(cls, x,geom_class, *args, **kwargs)
    
    def __init__(self,x,geom_class,*args,protein_data=None,proteins_list=None,**kwards):

        super().__init__(x,geom_class)
        if protein_data is not None:
            self.protein_data = protein_data
        else:
            self.protein_data = torch.empty(2,0,dtype=int)
            
        if proteins_list:
            self.proteins_list = proteins_list
        else:
            self.proteins_list = []
        
        
    def add_protein(self,protein,prot_origins=None):
        ref_index = protein.ref_pair.get_index()
        
        if prot_origins is None:
            prot_origins = prot.get_true_pos(prot.cg_structure.dna)
        prot_ori_frames = torch.tile(prot_origins,(self.shape[0],1,1))
        new_traj = torch.hstack([self[:,:ref_index+1],prot_ori_frames,self[:,ref_index+1:]])
        
        new_protein_data = torch.tensor([ref_index,protein.n_cg_beads],dtype=int).reshape(2,1)
        self.protein_data = torch.hstack([self.protein_data,new_protein_data])
        self.proteins_list.append(protein)
        
        return Origins_Tensor(new_traj,self.geom_class,protein_data = self.protein_data,proteins_list=self.proteins_list)
    
    def get_prot_traj_by_ref_index(self,ref_index):
        ref_index = self.__update_index(ref_index)
        stop = ref_index + self.protein_data[1][self.protein_data[0] == ref_index].sum()
        prot_traj =  super().__getitem__((slice(None, None, None), slice(ref_index, stop, None)))
        prot_traj.protein_data = torch.empty(2,0,dtype=int)
        prot_traj.proteins_list = []
        return prot_traj
    
    def __update_index(self,ind):
        return ind + self.protein_data[1][self.protein_data[0] < ind].sum()
    
        
    def __get_slice(self,frame_sl):
        if isinstance(frame_sl,slice):
            if frame_sl.start:
                start = self.__update_index(frame_sl.start)
            else:
                start = 0
            if frame_sl.stop:
                stop = self.__update_index(frame_sl.stop)
            else:
                stop = self.shape[1]
            
            
            # if frame_sl.step:
            #     check_step = (self.protein_data[0] - start)% frame_sl.step == 0
            #     included_proteins = (included_proteins + check_step) == 2
            #     step = frame_sl.step
            # else:
            #     step = 1
            return slice(start,stop)
        
        elif isinstance(frame_sl,int):
            
            return self.__update_index(frame_sl)
            
                
            
    def __setitem__(self, sl, value):
        if isinstance(sl,tuple):
            sl = list(sl)
            sl[1] = self.__get_slice(sl[1])
        super().__setitem__(sl,value)
        
    def __getitem__(self,sl):
        if isinstance(sl,tuple):
            sl = list(sl)
            sl[1] = self.__get_slice(sl[1])
        it = super().__getitem__(sl)
        it.protein_data = self.protein_data
        it.proteins_list = self.proteins_list
        return it