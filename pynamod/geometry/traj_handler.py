import torch
from operator import attrgetter

class Traj_Handler:
    def __init__(self,dtype,shapes,attrs,cur_st_cls='',traj_class=torch.tensor):
        for shape,attr in zip(shapes,attrs):
            setattr(self,f'_{attr}_traj',traj_class(torch.zeros(*shape,dtype=dtype)))
            
            
        self._cur_st_cls = attrgetter(cur_st_cls+'_cur_tr_step')
        self.test_sw = False
    
    def extend_trajectories(self,attrs,ln):
        for attr in attrs:
            traj_name = f'_{attr}_traj'
            cur_traj = getattr(self,traj_name)
            ext_traj = torch.vstack([cur_traj,torch.zeros(int(ln),*cur_traj.shape[1:],dtype=cur_traj.dtype)])
            if hasattr(cur_traj,'geom_class'):
                ext_traj.geom_class = cur_traj.geom_class
            setattr(self,traj_name,ext_traj)
            
    def _get_cur_step(self):
        if self.test_sw:
            cur_step = self._cur_st_cls(self)
        else:
            try:
                cur_step = self._cur_st_cls(self)
            except AttributeError:
                cur_step = 0
        return cur_step
        
    def setter_from_tr(self,value,attr):
        cur_step = self._get_cur_step()
        traj = getattr(self,f'_{attr}_traj')
        if traj.shape[0] > cur_step:
            traj[cur_step] = value
        else:
            new_traj = torch.cat([traj,value.reshape(1,*traj.shape[1:])])
            if hasattr(traj,'geom_class'):
                new_traj.geom_class = traj.geom_class
            setattr(self,f'_{attr}_traj',new_traj)
            
    def __getter_from_tr(self,attr):
        cur_step = self._get_cur_step()
        return getattr(self,f'_{attr}_traj')[cur_step]
        
    def get_property_from_tr(attr,fset=None,fget=None):
        if not fset:
            fset = lambda self,value: self.setter_from_tr(value,attr=attr)
        if not fget:
            fget = lambda self: self.__getter_from_tr(attr=attr)     
        return property(fget,fset)