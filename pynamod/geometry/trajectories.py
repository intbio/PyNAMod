import torch
import numpy as np
import h5py

from pynamod.geometry.tensor_subclasses import mod_Tensor,Origins_Tensor

class Trajectory:
    '''Class that processes trajectories of given parameters. It stores created trajectories for each given parameter and link to the class with current step of trajectories (could be self). All parameters are then should be defined as property using get_property_from_tr.'''
    def __init__(self,attrs_names):
        self.attrs_names = ['origins','ref_frames','local_params']
        if attrs_names:
            self.attrs_names += attrs_names

        self.cur_step = 0
        self.traj_step = 1

    def __iter__(self):
        self.cur_step = 0
        for i in range(len(self)):
            if self.cur_step > len(self) - 1:
                break
            yield self.cur_step
            self.cur_step += self.traj_step

        self.cur_step = 0

    origins = property(fset=lambda self, value: self._set_frame_attr('origins',value),
                       fget=lambda self: self._get_frame_attr('origins'))
    ref_frames = property(fset=lambda self, value: self._set_frame_attr('ref_frames',value),
                          fget=lambda self: self._get_frame_attr('ref_frames'))
    local_params = property(fset=lambda self, value: self._set_frame_attr('local_params',value),
                            fget=lambda self: self._get_frame_attr('local_params'))

    rlsp_origins = property(fset=lambda self, value: self._set_frame_attr('rlsp_origins',value),
                       fget=lambda self: self._get_frame_attr('rlsp_origins'))


class Tensor_Trajectory(Trajectory):
    def __init__(self,dtype,traj_len,data_len,traj_class,*traj_class_attrs,attrs_names=None,shapes=None):
        self.shapes = [(traj_len,data_len,1,3),(traj_len,data_len,3,3),(traj_len,data_len,6)]
        self.dtype = dtype
        self.data_len = data_len
        self.traj_class = traj_class
        self.traj_class_attrs = traj_class_attrs
        if shapes:
            self.shapes += shapes
        super().__init__(attrs_names)
        for shape,attr in zip(self.shapes,self.attrs_names):
            setattr(self,f'{attr}_traj',traj_class(torch.zeros(*shape,dtype=dtype),*traj_class_attrs))


    def copy(self,*traj_class_attrs):
        if not traj_class_attrs:
            traj_class_attrs = self.traj_class_attrs
        new = Tensor_Trajectory(bool,1,1,torch.tensor,attrs_names=self.attrs_names[3:],shapes=self.shapes[3:])
        for attr in self.attrs_names:
            setattr(new,f'{attr}_traj',self.traj_class(self.get_attr_trajectory(attr),*traj_class_attrs))

        return new

    def to(self,device):
        for attr in self.attrs_names:
            setattr(self,f'{attr}_traj',self.traj_class(self.get_attr_trajectory(attr)).to(device))

    def extend(self,other_traj=None,**values_to_extend):

        for attr in self.attrs_names:
            tensor = self.get_attr_trajectory(attr)
            if other_traj:
                value = other_traj.get_attr_trajectory(attr)
            else:
                value = values_to_extend[attr]
            setattr(self,f'{attr}_traj',torch.concat([tensor,torch.tensor(value)]))

    def add_attr(self,attr,shape):
        if attr in self.attrs:
            raise ValueError(f'Trajectory already has attribute {attr}')

        self.res_trajectory.attrs_names.append(attr)
        self.res_trajectory.shapes.append(shape)

        setattr(self,f'{attr}_traj',self.traj_class(torch.zeros((len(self),*shape),dtype=self.dtype),*self.traj_class_attrs))

    def get_attr_trajectory(self, attr):
        return getattr(self,attr+'_traj')

    def _create_frame(self):
        for attr,shape in zip(self.attrs_names,self.shapes):
            attr += '_traj'
            tensor = self.get_attr_trajectory(attr)
            setattr(self, attr, torch.concat([tensor,torch.zeros(*shape[1:])]))   

    def _get_frame_attr(self,attr,frame=None):
        if not frame:
            frame = self.cur_step
        return self.get_attr_trajectory(attr)[frame]

    def _set_frame_attr(self,attr,value,frame=None):
        if not frame:
            frame = self.cur_step
        if frame >= len(self):
            self._create_frame()
        self.get_attr_trajectory(attr)[frame] = value

    def __len__(self):
        return self.origins_traj.shape[0]

    def __getitem__(self, sl):

        traj_len, data_len = self.shapes[0][:2]
        new = Tensor_Trajectory(self.dtype,traj_len,data_len,self.traj_class,*self.traj_class_attrs)
        new.shapes = self.shapes
        new.attrs_names = self.attrs_names

        for attr in self.attrs_names:
            setattr(new,f'{attr}_traj',self.traj_class(self.get_attr_trajectory(attr)[sl],*self.traj_class_attrs))

        return new


class H5_Trajectory(Trajectory):
    def __init__(self,filename,data_len,mode='r',attrs_names=None,shapes=None,string_format_val=5,**kwards):
        if shapes:
            self.shapes = shapes
        else:
            self.shapes = [(data_len,1,3),(data_len,3,3),(data_len,6)]
        super().__init__(attrs_names)
        self.file = h5py.File(filename, mode, libver='latest')
        self._dataset_kwards = kwards
        self.string_format_val = string_format_val

        if mode in ('w','x','w-'):
            self._last_frame_ind = -1

        elif mode == 'r':
            self._last_frame_ind = len(self.file) - 1

        elif mode in ('r+','a'):
            self._last_frame_ind = self.cur_step = len(self.file) - 1

        if len(self) != 0:
            for k in self.file[str(self._last_frame_ind).zfill(self.string_format_val)].keys():
                if k not in self.attrs_names:
                    self.attrs_names += [str(k)]
                    self.shapes += self.file[str(self._last_frame_ind).zfill(self.string_format_val)][str(k)][:].shape

    def extend(self,other_traj=None,**values_to_extend):
        if other_traj:
            for step in other_traj:
                attrs = {}
                for attr in self.attrs_names:
                    attrs[attr] = other_traj._get_frame_attr(attr)
                self.add_frame(self._last_frame_ind+1,**attrs)

        else:
            for i in range(values_to_extend['origins_traj'].shape[0]):
                attrs = {}
                for attr in self.attrs_names:
                    attrs[attr] = values_to_extend[attr+'_traj']

                self.add_frame(self._last_frame_ind+1,**attrs)

    def add_attr(self,attr,shape):
        if attr in self.attrs_names:
            raise ValueError(f'Trajectory already has attribute {attr}')

        self.attrs_names.append(attr)
        self.shapes.append(shape)

        for i in range(len(self)):
            self.file[str(i).zfill(self.string_format_val)].create_dataset(attr,shape=shape,**self._dataset_kwards)

    def add_frame(self,step,**attrs):
        if step > self._last_frame_ind:
            self._create_frame(step)
        for attr,value in attrs.items():

            self.file[str(self._last_frame_ind).zfill(self.string_format_val)][attr][:] = value

    def copy(self,new):
        return self


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
        return torch.from_numpy(self.file[str(frame).zfill(self.string_format_val)][attr][:])

    def _set_frame_attr(self,attr,value,frame=None):
        if not frame:
            frame = self.cur_step
        if frame > self._last_frame_ind:
            self._create_frame(frame)
        self.file[str(frame).zfill(self.string_format_val)][attr][:] = value

    def __len__(self):
        return self._last_frame_ind + 1

    def __del__(self):
        self.file.close()

    def get_attr_trajectory(self,attr,start=0,stop=None,step=1):
        attr_traj = np.zeros((self._last_frame_ind+1,*self.shapes[self.attrs_names.index(attr)]))
        if stop is None:
            stop = self._last_frame_ind + 1
        for i in range(start,stop,step):
            attr_traj[i] = self.file[str(i).zfill(self.string_format_val)][attr]
            
        return attr_traj
    
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