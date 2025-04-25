import torch

from pynamod.geometry.trajectories import Tensor_Trajectory,H5_Trajectory
from pynamod.geometry.bp_step_geometry import Geometry_Functions
from pynamod.geometry.bp_step_geometry import Geometry_Functions
from pynamod.geometry.tensor_subclasses import mod_Tensor

class Geometrical_Parameters(Geometry_Functions):
    def __init__(self,local_params = None, ref_frames = None, origins = None,trajectory=None,traj_len=1,pair_params=False,auto_rebuild=True,empty=False):

        self.pair_params = pair_params
        
        self._auto_rebuild_sw = self.auto_rebuild = auto_rebuild
        

        if local_params is not None:
            self.dtype = local_params.dtype
            self.len = local_params.shape[0]

        if origins is not None:
            self.dtype = origins.dtype
            self.len = origins.shape[0]
            origins = origins.reshape(-1,1,3)
            
        if trajectory:
            self.trajectory = trajectory
            if isinstance(trajectory,H5_Trajectory):
                self.auto_rebuild = False
        elif not empty:
            self.trajectory = Tensor_Trajectory(self.dtype,traj_len,self.len,lambda x: mod_Tensor(x,self))
        if not empty:
            self.get_new_params_set(local_params, ref_frames, origins)
        
    
    def to(self,device):
        self.trajectory.ref_frames_traj = mod_Tensor(self.trajectory.ref_frames_traj.to(device),self)
        self.trajectory.origins_traj = mod_Tensor(self.trajectory.origins_traj.to(device),self)
        self.trajectory.local_params_traj = mod_Tensor(self.trajectory.local_params_traj.to(device),self)
    
    
    def get_new_params_set(self,local_params = None, ref_frames = None, origins = None):
        set_from_local_params = local_params is not None
        set_from_r_and_o = ref_frames is not None and origins is not None
        
        self._auto_rebuild_sw = False 
        
        if set_from_r_and_o and set_from_local_params:
            
            if not origins.dtype == ref_frames.dtype == local_params.dtype:
                raise TypeError("Dtypes don't match")
                
            self.__set_from_all_params(local_params,ref_frames,origins)

        elif set_from_r_and_o:
            
            if origins.dtype != ref_frames.dtype:
                raise TypeError("Origins and reference frames dtypes don't match")
            
            self.__set_from_r_and_o(ref_frames,origins)
            
        elif set_from_local_params:
            
            self.__set_from_local_params(local_params)
            
        else:
            raise TypeError('Geometrical_parameters should be initialized with local parameters or reference frames and origins')
       
        self._auto_rebuild_sw = self.auto_rebuild
        return self
    
    
    def __set_from_all_params(self,local_params,ref_frames,origins):
        for attr in ('ref_frames','origins','local_params'):
            tens = locals()[attr]
            setattr(self,attr,tens)
            
    def __set_from_r_and_o(self,ref_frames,origins):
        self.ref_frames = ref_frames
        self.origins = origins    
        self.local_params = torch.zeros(self.len,6,dtype=self.dtype)
        self.rebuild('rebuild_local_params') 
        
    def __set_from_local_params(self,local_params):
        self.local_params = local_params
        self.ref_frames = torch.zeros((self.len,3,3),dtype=self.dtype)
        self.origins = torch.zeros((self.len,1,3),dtype=self.dtype)
        self.rebuild('rebuild_ref_frames_and_ori')
        
    def copy(self):
        new = Geometrical_Parameters(empty=True,pair_params=self.pair_params)
        new.trajectory = self.trajectory.copy(new)
        return new
    
    
    def rebuild(self,rebuild_func_name,*args,**kwards):
        self._auto_rebuild_sw = False

        getattr(self,rebuild_func_name)(*args,**kwards)
        
        self._auto_rebuild_sw = self.auto_rebuild
        
        
        
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
        self.trajectory._set_frame_attr(attr,value)
        if self._auto_rebuild_sw:
            self.rebuild(rebuild_func_name)
            
    def __getter(self,attr):
        return self.trajectory._get_frame_attr(attr)
    
    def __set_property(attr,rebuild_func_name):
        setter = lambda self,value: self.__setter(value,attr = attr,
                                    rebuild_func_name=rebuild_func_name)
        getter = lambda self: self.__getter(attr=attr)
        
        return property(fset=setter,fget=getter)
        
    local_params = __set_property('local_params','rebuild_ref_frames_and_ori')
    ref_frames = __set_property('ref_frames','rebuild_local_params')
    origins = __set_property('origins','rebuild_local_params')
    
            