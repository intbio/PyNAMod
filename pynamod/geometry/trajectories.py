import torch
import h5py



class Trajectory:
    '''Class that processes trajectories of given parameters. It stores created trajectories for each given parameter and link to the class with current step of trajectories (could be self). All parameters are then should be defined as property using get_property_from_tr.'''
    def __init__(self,traj_len,data_len,attrs_names,shapes):
        if shapes:
            self.shapes = shapes
        else:
            self.shapes = [(traj_len,data_len,1,3),(traj_len,data_len,3,3),(traj_len,data_len,6)]
        
        if attrs_names:
            self.attrs_names = attrs_names
        else:
            self.attrs_names = ['origins','ref_frames','local_params']
            
        self.cur_step = 0  
        

class Tensor_Trajectory(Trajectory):
    def __init__(self,dtype,traj_len,data_len,geom_class,attrs_names=None,shapes=None,**kwards):
        super().__init__(traj_len,data_len,attrs_names,shapes)
        for shape,attr in zip(self.shapes,self.attrs_names):
            if kwards:
                setattr(self,f'{attr}_traj',mod_Tensor(kwards[f'{attr}_traj'],geom_class))
            else:    
                setattr(self,f'{attr}_traj',mod_Tensor(torch.zeros(*shape,dtype=dtype),geom_class))
            
    
    def copy(self,geom_class):
        return Tensor_Trajectory(None,None,None,geom_class,origins_traj=self.origins_traj,
                                    ref_frames_traj=self.ref_frames_traj,local_params_traj=self.local_params_traj)
            
    def _get_frame_attr(self,attr,frame=None):
        if not frame:
            frame = self.cur_step
        return getattr(self,attr)[frame]
    
    def _set_frame_attr(self,attr,value,frame=None):
        if not frame:
            frame = self.cur_step
        getattr(self,attr)[frame] = value
        

class H5_Trajectory(Trajectory):
    def __init__(self,filename,initial_len,data_len,attrs_names=None,shapes=None,**kwards):
        super().__init__(initial_len,data_len,attrs_names,shapes)
        self.file = h5py.File(filename,'w')
        self._dataset_kwards = kwards
        self._last_frame_ind = -1
        for i in range(initial_len):
            self._create_frame(i)
            
    
    def add_frame(self,step,**attrs):
        if step > self._last_frame_ind:
            self._create_frame(step)
        for attr,value in attrs.items():
            self.file[str(self.cur_step)][f'{attr}'][:] = value


    def _create_frame(self,frame_ind):
        if self._last_frame_ind < frame_ind:
            self._last_frame_ind = frame_ind
            frame_ind = str(frame_ind)
            frame = self.file.create_group(frame_ind)
            for attr_name,shape in zip(self.attrs_names,self.shapes):
                ds = frame.create_dataset(attr_name,shape=shape,**self._dataset_kwards)
        else:
            raise ValueError('frame creation failed, frame already exists')

    def _get_frame_attr(self,attr,frame=None):
        if not frame:
            frame = self.cur_step
        return self.file[str(frame)][attr]
    
    def _set_frame_attr(self,attr,value,frame=None):
        if not frame:
            frame = self.cur_step
        if frame > self._last_frame_ind:
            self._create_frame(frame)
        self.file[str(frame)][attr] = value

            
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
                    