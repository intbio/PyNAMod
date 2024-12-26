import torch
from operator import attrgetter

class Traj_Handler:
    def __init__(self,dtype,shapes,attrs,cur_st_cls=''):
        for shape,attr in zip(shapes,attrs):
            setattr(self,f'_{attr}_traj',torch.zeros(*shape,dtype=dtype))
            
            
        self._cur_st_cls = attrgetter(cur_st_cls+'_cur_tr_step')
        self.test_sw = False
            
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