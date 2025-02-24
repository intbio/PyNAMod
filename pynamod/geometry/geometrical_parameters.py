import torch

from pynamod.geometry.traj_handler import Traj_Handler
from pynamod.geometry.bp_step_geometry import Geometry_Functions

class Geometrical_Parameters(Traj_Handler,Geometry_Functions):
    def __init__(self,local_params = None, ref_frames = None, origins = None,pair_params=False,auto_rebuild=True,traj_len=None,cur_step_cls = ''):

        self.pair_params = pair_params
        
        self._auto_rebuild_sw = self.auto_rebuild = auto_rebuild
        
        if not traj_len:
            traj_len = 1
        if not cur_step_cls:
            #not fixed?, cur_tr_step should be taken from cur_step_cls if given
            self._cur_tr_step = 0
            self.cur_step_cls = ''
        
        local_params_shape = None
        if local_params is not None:
            self.dtype = local_params.dtype
            self.len = local_params.shape[0]
            local_params_shape = (traj_len,*local_params.shape)
        if origins is not None:
            self.dtype = origins.dtype
            self.len = origins.shape[0]
            origins = origins.reshape(-1,1,3)
            origins_shape = (traj_len,*origins.shape)
        elif self.len:
            origins_shape = (traj_len,self.len,1,3)
            
        if ref_frames is not None:
            ref_frames_shape = (traj_len,*ref_frames.shape)
        elif self.len:
            ref_frames_shape = (traj_len,self.len,3,3)
            
        if not local_params_shape:
            local_params_shape = (traj_len,self.len,6)
            
        
        
    
        super().__init__(self.dtype,[ref_frames_shape,origins_shape,local_params_shape],
                         ('ref_frames','origins','local_params'),cur_step_cls,traj_class=lambda o: mod_Tensor(o,self))
        
        self.get_new_params_set(local_params, ref_frames, origins)
        
    
    def to(self,device):
        self._ref_frames_traj = mod_Tensor(self._ref_frames_traj.to(device),self)
        self._origins_traj = mod_Tensor(self._origins_traj.to(device),self)
        self._local_params_traj = mod_Tensor(self._local_params_traj.to(device),self)
    
    def get_new_params_set(self,local_params = None, ref_frames = None, origins = None):
        init_from_local_params = local_params is not None
        init_from_r_and_o = ref_frames is not None and origins is not None
        
        self._auto_rebuild_sw = False    
        if init_from_r_and_o and init_from_local_params:     
            if not origins.dtype == ref_frames.dtype == local_params.dtype:
                raise TypeError("Dtypes don't match")
            for attr in ('ref_frames','origins','local_params'):
                tens = locals()[attr]
                if tens.shape == getattr(self,f'_{attr}_traj').shape:
                    setattr(self,f'_{attr}_traj',tens)
                else:
                    setattr(self,attr,tens)

        elif init_from_r_and_o:
            if origins.dtype != ref_frames.dtype:
                raise TypeError("Origins and reference frames dtypes don't match")
            
            self.ref_frames = ref_frames
            self.origins = origins    
            self.local_params = torch.zeros(self.len,6,dtype=self.dtype)
            self.rebuild('rebuild_local_params')
            
        elif init_from_local_params:
            self.local_params = local_params
            self.ref_frames = torch.zeros((self.len,3,3),dtype=self.dtype)
            self.origins = torch.zeros((self.len,1,3),dtype=self.dtype)
            self.rebuild('rebuild_ref_frames_and_ori')
        
        else:
            raise TypeError('Geometrical_parameters should be initialized with local parameters or reference frames and origins')
       
        self._auto_rebuild_sw = self.auto_rebuild
        return self
        
    def copy(self):
        return Geometrical_Parameters(local_params = self.local_params.clone(), ref_frames = self.ref_frames.clone(),
                                      origins = self.origins.clone(),pair_params=self.pair_params)
    
    
    def rebuild(self,rebuild_func_name,*args,**kwards):
        self._auto_rebuild_sw = False

        getattr(self,rebuild_func_name)(*args,**kwards)
        
        self._auto_rebuild_sw = self.auto_rebuild
        
        
    def traj_len(self):
        return self._local_params_traj.shape[0]
        
        
    def __getitem__(self,sl):
        neg_step = False
        if isinstance(sl,slice):
            if sl.step is not None:
                if sl.step < 0:
                    sl = slice(sl.start,sl.stop,-1*sl.step)
                    neg_step = True
        it = Geometrical_Parameters(ref_frames = self.ref_frames[sl],
                                      origins = self.origins[sl],pair_params=self.pair_params)
        if neg_step:
            it._auto_rebuild_sw = False
            it.origins = it.origins.flip(dims=(0,))
            it._auto_rebuild_sw = it.auto_rebuild
            it.ref_frames = it.ref_frames.flip(dims=(0,))
            it.local_params *= -1
        
        return it
        
        
        
    
    
    def __setter(self,value,attr,rebuild_func_name):
        self.setter_from_tr(mod_Tensor(value,self),attr)
        if self._auto_rebuild_sw:
            self.rebuild(rebuild_func_name)

        
    local_params = Traj_Handler.get_property_from_tr('local_params',
                                                     fset=lambda self,value: self.__setter(value,attr = 'local_params',
                                                                        rebuild_func_name='rebuild_ref_frames_and_ori'))
    ref_frames = Traj_Handler.get_property_from_tr('ref_frames',
                                                     fset=lambda self,value: self.__setter(value,attr = 'ref_frames',
                                                                        rebuild_func_name='rebuild_local_params'))
    origins = Traj_Handler.get_property_from_tr('origins',
                                                     fset=lambda self,value: self.__setter(value.reshape(-1,1,3),attr = 'origins',
                                                                        rebuild_func_name='rebuild_local_params'))
    

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
                if self.shape[-1] == 3:
                    self.geom_class.rebuild('rebuild_local_params')
                elif self.shape[-1] == 6:
                    self.geom_class.rebuild('rebuild_ref_frames_and_ori',rebuild_proteins=True)

            elif isinstance(sl,int) or (isinstance(sl,torch.Tensor) and sl.shape == tuple()):
                if self.shape[-1] == 3:
                    self.geom_class.rebuild('rebuild_local_params',start_index=sl)
                elif self.shape[-1] == 6:
                    self.geom_class.rebuild('rotate_ref_frames_and_ori',sl)
                
            