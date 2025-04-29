import torch
import numpy as np
import h5py

from pynamod.geometry.tensor_subclasses import mod_Tensor,Origins_Tensor

class Trajectory:
    '''Class that processes trajectories of given parameters. It stores created trajectories for each given parameter and link to the class with current step of trajectories (could be self). All parameters are then should be defined as property using get_property_from_tr.'''
    def __init__(self,traj_len,data_len,attrs_names):
        self.attrs_names = ['origins','ref_frames','local_params']
        if attrs_names:
            self.attrs_names += attrs_names
            
        self.cur_step = 0  
        

class Tensor_Trajectory(Trajectory):
    def __init__(self,dtype,traj_len,data_len,traj_class,attrs_names=None,shapes=None):
        self.shapes = [(traj_len,data_len,1,3),(traj_len,data_len,3,3),(traj_len,data_len,6)]
        self.traj_class = traj_class
        if shapes:
            self.shapes += shapes
        super().__init__(traj_len,data_len,attrs_names)
        for shape,attr in zip(self.shapes,self.attrs_names):
            setattr(self,f'{attr}_traj',traj_class(torch.zeros(*shape,dtype=dtype)))
                
    
    def copy(self,geom_class):
        new = Tensor_Trajectory(bool,1,1,torch.tensor,attrs_names=self.attrs_names[3:],shapes=self.shapes[3:])
        for attr in self.attrs_names:
            setattr(new,f'{attr}_traj',mod_Tensor(getattr(self,f'{attr}_traj'),geom_class))
        
        return new
    
    def to(self,device):
        for attr in self.attrs_names:
            setattr(self,f'{attr}_traj',self.traj_class(getattr(self,f'{attr}_traj').to(device)))
            
    def _get_frame_attr(self,attr,frame=None):
        if not frame:
            frame = self.cur_step
        attr += '_traj'
        return getattr(self,attr)[frame]
    
    def _set_frame_attr(self,attr,value,frame=None):
        if not frame:
            frame = self.cur_step

        attr += '_traj'
        getattr(self,attr)[frame] = value
        
    def get_len(self):
        return self.origins_traj.shape[0]
        

class H5_Trajectory(Trajectory):
    def __init__(self,filename,initial_len,data_len,mode='w',attrs_names=None,shapes=None,string_format_val=5,**kwards):
        if shapes:
            self.shapes = shapes
        else:
            self.shapes = [(data_len,1,3),(data_len,3,3),(data_len,6)]
        super().__init__(initial_len,data_len,attrs_names)
        self.file = h5py.File(filename,mode)
        self._dataset_kwards = kwards
        self.string_format_val = string_format_val
        if mode == 'w':
            self._last_frame_ind = -1
            for i in range(initial_len):
                self._create_frame(i)
        elif mode == 'r':
            self._last_frame_ind = len(self.file)
            
        elif mode == 'r+':
            self._last_frame_ind = self.cur_step = init_ln = len(self.file) - 1
            for i in range(1,initial_len+1):
                self._create_frame(init_ln+i)
            

    
    def add_frame(self,step,**attrs):
        if step > self._last_frame_ind:
            self._create_frame(step)
        for attr,value in attrs.items():
            self.file[str(self.cur_step).zfill(self.string_format_val)][attr][:] = value


    def _create_frame(self,frame_ind):
        if self._last_frame_ind < frame_ind:
            self._last_frame_ind = frame_ind
            frame_ind = str(frame_ind).zfill(self.string_format_val)
            frame = self.file.create_group(frame_ind)
            for attr_name,shape in zip(self.attrs_names,self.shapes):
                ds = frame.create_dataset(attr_name,shape=shape,**self._dataset_kwards)
        else:
            raise KeyError('frame creation failed, frame already exists')

    def _get_frame_attr(self,attr,frame=None):
        if not frame:
            frame = self.cur_step
        return self.file[str(frame).zfill(self.string_format_val)][attr][:]
    
    def _set_frame_attr(self,attr,value,frame=None):
        if not frame:
            frame = self.cur_step
        if frame > self._last_frame_ind:
            self._create_frame(frame)
        self.file[str(frame).zfill(self.string_format_val)][attr][:] = value
        
    def get_len(self):
        return self._last_frame_ind
    
    def get_energies_arr(self,ind):
        if not 'energies' in self.attrs_names: return None
    
        energy_arr = np.zeros((self._last_frame_ind))
        for i in range(self._last_frame_ind):
            energy_arr[i] = self.file[str(i).zfill(self.string_format_val)]['energies'][ind]
        
        return energy_arr
    
    def get_energy_array_slice(self,name,sl):
        if name != 'total':
            ind  = ['bend','elst','ld','restr'].index(name)
            
        energy_arr = np.zeros((self._last_frame_ind))[sl]
        for en_i,i in enumerate(list(range(self._last_frame_ind))[sl]):
            if name == 'total':
                energy_arr[en_i] = np.sum(self.file[str(i).zfill(self.string_format_val)]['energies'][:])
            else:
                energy_arr[en_i] = self.file[str(i).zfill(self.string_format_val)]['energies'][ind]
                
        return energy_arr
        
        
    
    bend_energies = property(fget=lambda self:self.get_energies_arr(0))
    elst_energies = property(fget=lambda self:self.get_energies_arr(1))
    ld_energies = property(fget=lambda self:self.get_energies_arr(2))
    restr_energies = property(fget=lambda self:self.get_energies_arr(3))
    
    @property
    def total_energies(self):
        if not 'energies' in self.attrs_names: return None
        return self.bend_energies + self.elst_energies + self.ld_energies + self.restr_energies
    
                                                       
class Integrator_Trajectory(Tensor_Trajectory):
    def __init__(self,proteins_list,dtype,traj_len,data_len):
        if proteins_list:
            self.prot_origins_ln = sum([protein.n_cg_beads for protein in proteins_list])
            proteins_data = [[protein.n_cg_beads,protein.ref_pair.ind] for protein in proteins_list]
            self.proteins_data = torch.tensor(proteins_data,dtype=int).T
        else:
            self.prot_origins_ln = 0
            self.proteins_data = None
        super().__init__(dtype,traj_len,data_len,torch.tensor)
        self.data_len = data_len
        self.origins_traj = torch.zeros(traj_len,data_len+self.proteins_data[0].sum().item(),1,3,dtype=dtype)
        
        
    def get_proteins_slice_ind(self,dna_index):
        if self.proteins_data is not None:
            return self.proteins_data[0][self.proteins_data[1] >= dna_index].sum() + self.data_len
        else:
            return self.data_len
        

        
    origins = property(fset=lambda self,value: self._set_frame_attr('origins',value),
                       fget=lambda self: self._get_frame_attr('origins'))
    ref_frames = property(fset=lambda self,value: self._set_frame_attr('ref_frames',value),
                          fget=lambda self: self._get_frame_attr('ref_frames'))
    local_params = property(fset=lambda self,value: self._set_frame_attr('local_params',value),
                            fget=lambda self: self._get_frame_attr('local_params'))
    