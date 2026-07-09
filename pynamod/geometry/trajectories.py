import torch
import numpy as np
import h5py
from abc import ABC,abstractmethod

class Trajectory(ABC):
    '''Class that processes trajectories of given parameters. It stores created trajectories for each given parameter and link to the class with current step of trajectories (could be self). All parameters are then should be defined as property using get_property_from_tr.'''
    def __init__(self,attrs_names):
        self.attrs_names = ['origins', 'ref_frames', 'local_params']
        if attrs_names:
            self.attrs_names += attrs_names

        self.cur_step = 0
        self.traj_step = 1

    def __iter__(self):
        for i in range(0,len(self),self.traj_step):
                
            self.cur_step = i
            if self.origins.sum() == 0:
                break
            yield self.cur_step

        self.cur_step = 0

    def add_frames(self, start_step=None, end_step=None,
                   other_traj=None, **attrs_trajectories):
        if other_traj:
            start_step = len(self)
            end_step = len(other_traj)
        if start_step is None:
            start_step = len(self)
        if end_step is None:
            end_step = start_step + 1
        if end_step >= len(self):
            self._create_frames(end_step-start_step)
        for attr in self.attrs_names:
            self._set_new_frames(attr, start_step, end_step,
                                 other_traj, attrs_trajectories)
        self.cur_step += end_step - start_step - 1

    def add_attr(self, attr, shape):
        if attr in self.attrs_names:
            raise ValueError(f'Trajectory already has attribute {attr}')

        self.attrs_names.append(attr)
        self.shapes.append(shape)

        self._create_attr_trajectory(attr, shape)

    @abstractmethod
    def _create_attr_trajectory(self, attr, shape):
        pass

    @abstractmethod
    def _set_new_frames(self, attr, start_step, end_step,
                        other_traj, attrs_trajectories):
        pass

    origins = property(fset=lambda self, value: self._set_frame_attr('origins',value),
                       fget=lambda self: self._get_frame_attr('origins'))
    ref_frames = property(fset=lambda self, value: self._set_frame_attr('ref_frames',value),
                          fget=lambda self: self._get_frame_attr('ref_frames'))
    local_params = property(fset=lambda self, value: self._set_frame_attr('local_params',value),
                            fget=lambda self: self._get_frame_attr('local_params'))

    rlsp_origins = property(fset=lambda self, value: self._set_frame_attr('rlsp_origins',value),
                       fget=lambda self: self._get_frame_attr('rlsp_origins'))

    energies = property(fset=lambda self, value: self._set_frame_attr('energies',value),
                       fget=lambda self: self._get_frame_attr('energies'))


class Tensor_Trajectory(Trajectory):
    def __init__(self, dtype, traj_len, data_len, traj_class, *traj_class_attrs,
                 attrs_names=None, shapes=None):
        self.shapes = [(data_len, 1, 3), (data_len, 3, 3), (data_len, 6)]
        self.dtype = dtype
        self.traj_len = traj_len
        self.data_len = data_len
        self.traj_class = traj_class
        self.traj_class_attrs = traj_class_attrs
        if shapes:
            self.shapes += shapes
        super().__init__(attrs_names)
        for shape, attr in zip(self.shapes, self.attrs_names):
            setattr(self, f'{attr}_traj', traj_class(torch.zeros(traj_len, *shape, dtype=dtype), *traj_class_attrs))

    def copy(self, *traj_class_attrs):
        if not traj_class_attrs:
            traj_class_attrs = self.traj_class_attrs
        new = Tensor_Trajectory(self.dtype, len(self), self.data_len,
                                self.traj_class, traj_class_attrs,
                                attrs_names=self.attrs_names[3:], shapes=self.shapes[3:])
        for attr in self.attrs_names:
            setattr(new, f'{attr}_traj', new.traj_class(self.get_attr_trajectory(attr), *traj_class_attrs))

        return new

    def to(self, device):
        for attr in self.attrs_names:
            setattr(self, f'{attr}_traj', self.traj_class(self.get_attr_trajectory(attr)).to(device))

    def _set_new_frames(self, attr, start_step, end_step,
                        other_traj, attr_trajectory):
        if other_traj:
            value = other_traj.get_attr_trajectory(attr)
        else:
            value = torch.tensor(
                attr_trajectory[attr + '_traj'])
        self.get_attr_trajectory(attr)[start_step:end_step] = value

    def _create_attr_trajectory(self, attr, shape):
        setattr(self, attr+'_traj',
                self.traj_class(
                            torch.zeros((len(self), *shape),
                            dtype=self.dtype),
                *self.traj_class_attrs))

    def get_attr_trajectory(self, attr, start=0, stop=None, step=1):
        return getattr(self, attr+'_traj')[start:stop:step]

    def _create_frames(self, ln=1):
        for attr, shape in zip(self.attrs_names, self.shapes):
            tensor = self.get_attr_trajectory(attr)
            setattr(self, attr+'_traj', torch.concat([tensor, torch.zeros(ln,*shape)]))
        self.traj_len += ln

    def _get_frame_attr(self, attr, frame=None):
        if not frame:
            frame = self.cur_step
        return self.get_attr_trajectory(attr)[frame]

    def _set_frame_attr(self, attr, value, frame=None):
        if not frame:
            frame = self.cur_step
        if frame >= len(self):
            self._create_frames()
        self.get_attr_trajectory(attr)[frame] = value

    def __len__(self):
        return self.origins_traj.shape[0]

    def __getitem__(self, sl):

        new = Tensor_Trajectory(self.dtype, self.traj_len, self.data_len, self.traj_class, *self.traj_class_attrs)
        new.shapes = self.shapes
        new.attrs_names = self.attrs_names

        for attr in self.attrs_names:
            setattr(new, f'{attr}_traj', self.traj_class(self.get_attr_trajectory(attr)[sl], *self.traj_class_attrs))

        return new


class H5_Trajectory(Trajectory):
    def __init__(self, filename, data_len, mode='r', attrs_names=None,
                 shapes=None, string_format_val=5, chunk_size=10, **kwards):
        if shapes:
            self.shapes = shapes
        else:
            self.shapes = [(data_len, 1, 3), (data_len, 3, 3), (data_len, 6)]
        super().__init__(attrs_names)
        self.file = h5py.File(filename, mode, libver='latest')
        self._dataset_kwards = kwards
        self.string_format_val = string_format_val
        self.chunk_size = chunk_size

        if mode in ('w', 'x', 'w-'):
            self.total_len = 0

        elif mode == 'r':
            self.chunk_size = self._get_group(0)['origins'].shape[0]
            self.total_len = len(self.file.keys())*self.chunk_size

        elif mode in ('r+', 'a'):
            self.total_len = len(self.file)
            self.cur_step = self.total_len - 1
        if len(self) != 0:
            for k in self._get_group(0).keys():
                if k not in self.attrs_names:
                    self.attrs_names += [str(k)]
                    self.shapes += self._get_group(0)[str(k)].shape


    def _create_attr_trajectory(self, attr, shape):
        for i in range(len(self)//self.chunk_size):
            self._get_group(i).create_dataset(attr,
                                              shape=(self.chunk_size, *shape),
                                              **self._dataset_kwards)

    def _set_new_frames(self, attr, start_step, end_step,
                        other_traj, attr_trajectory):

        if other_traj:
            start_step = self.cur_step
            chunk_size = other_traj.chunk_size if hasattr(other_traj,'chunk_size') else len(other_traj)
            end_step = start_step + chunk_size
            for i in range(len(other_traj)//chunk_size):
                self._set_new_frames(attr,start_step,end_step,None,
                    {attr + '_traj':other_traj.get_attr_trajectory(attr, start=start_step, stop=end_step)})
                start_step = end_step
                end_step += chunk_size

        else:
            start_chunk, start_chunk_step = self._get_chunk_indices(start_step)
            end_chunk, end_chunk_step = self._get_chunk_indices(end_step)
            if end_chunk_step == 0:
                end_chunk -= 1
                end_chunk_step = self.chunk_size
            cur_end_chunk_step = self.chunk_size
            traj_start = 0
            if start_chunk != end_chunk:
                traj_stop = self.chunk_size - start_chunk_step
            else:
                traj_stop = end_chunk_step - start_chunk_step

            for chunk_ind in range(start_chunk, end_chunk+1):
                group = self._get_group(chunk_ind)
                if chunk_ind == end_chunk:
                    cur_end_chunk_step = end_chunk_step
                group[attr][start_chunk_step:cur_end_chunk_step] = attr_trajectory[attr + '_traj'][traj_start:traj_stop]
                start_chunk_step = 0
                traj_start = traj_stop
                traj_stop += self.chunk_size

    def copy(self, new):
        return self

    def _create_frames(self, ln=1):
        added_ln = 0
        for chunk_ind in range(len(self)//self.chunk_size,
                               (len(self) + ln)//self.chunk_size + 1):

            name = self._get_group_name(chunk_ind)
            new_chunk = self.file.create_group(name)
            for attr_name, shape in zip(self.attrs_names, self.shapes):
                ds = new_chunk.create_dataset(attr_name,
                        shape=(self.chunk_size, *shape), **self._dataset_kwards)
            added_ln += self.chunk_size
        self.total_len += added_ln

    def _get_group_name(self, ind):
        return str(ind).zfill(self.string_format_val)

    def _get_group(self, ind):
        return self.file[self._get_group_name(ind)]

    def _get_chunk_indices(self, frame):
        chunk = frame // self.chunk_size
        chunk_step = frame % self.chunk_size
        return chunk, chunk_step

    def _get_frame_attr(self, attr, frame=None):
        if frame is None:
            frame = self.cur_step
        chunk, chunk_step = self._get_chunk_indices(frame)
        return torch.from_numpy(self._get_group(chunk)[attr][chunk_step])

    def _set_frame_attr(self, attr, value, frame=None):
        if frame is None:
            frame = self.cur_step
        if frame >= len(self):
            self._create_frames()
        chunk, chunk_step = self._get_chunk_indices(frame)
        self._get_group(chunk)[attr][chunk_step] = value

    def __len__(self):
        return self.total_len

    def get_attr_trajectory(self, attr, start=0, stop=None, step=1):
        if stop is None:
            stop = len(self)
        attr_traj = np.zeros((stop//step, *self.shapes[self.attrs_names.index(attr)]))
        start_chunk, start_chunk_step = self._get_chunk_indices(start)
        end_chunk, end_chunk_step = self._get_chunk_indices(stop)
        cur_end_chunk_step = self.chunk_size
        traj_start = start
        traj_stop = self.chunk_size - traj_start
        for chunk_ind in range(start_chunk, end_chunk+1):
            attr_traj[traj_start:traj_stop] = self._get_group(chunk_ind)[start_chunk_step:cur_end_chunk_step]
            start_chunk_step = 0
            if chunk_ind == end_chunk:
                cur_end_chunk_step = end_chunk_step
            traj_start = traj_stop
            traj_stop += cur_end_chunk_step

        return attr_traj

    def close(self):
        self._finalizer()

    def get_energy_array_slice(self, name, sl):
        #Fix
        if name != 'total':
            ind = ['bend', 'elst', 'ld', 'restr'].index(name)

        energy_arr = np.zeros(len(self))[sl]
        for en_i,i in enumerate(list(range(len(self)))[sl]):
            if name == 'total':
                energy_arr[en_i] = np.sum(self._get_group(i)['energies'][:])
            else:
                energy_arr[en_i] = self._get_group(i)['energies'][ind]

        return energy_arr

    bend_energies = property(fget=lambda self:self.get_energies_arr(0))
    elst_energies = property(fget=lambda self:self.get_energies_arr(1))
    ld_energies = property(fget=lambda self:self.get_energies_arr(2))
    restr_energies = property(fget=lambda self:self.get_energies_arr(3))

    @property
    def total_energies(self):
        if not 'energies' in self.attrs_names: return None
        return self.bend_energies + self.elst_energies + self.ld_energies + self.restr_energies