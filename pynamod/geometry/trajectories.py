import h5py
import numpy as np
import torch

from pynamod.geometry.tensor_subclasses import Origins_Tensor, mod_Tensor


def open_h5(filename, mode='r', **kwargs):
    '''Open an HDF5 file; disable POSIX locking when supported (h5py >= 3.5).

    Network filesystems (NFS, Lustre, etc.) often fail synchronous locks with
    BlockingIOError (errno 11). For h5py < 3.5, set HDF5_USE_FILE_LOCKING=FALSE
    in the environment before opening files.
    '''
    kwargs.setdefault('locking', False)
    try:
        return h5py.File(filename, mode, **kwargs)
    except TypeError:
        kwargs.pop('locking', None)
        return h5py.File(filename, mode, **kwargs)


class Trajectory:
    '''Class that processes trajectories of given parameters. It stores created trajectories for each given parameter and link to the class with current step of trajectories (could be self). All parameters are then should be defined as property using get_property_from_tr.'''

    def __init__(self, attrs_names):
        self.attrs_names = ['origins', 'ref_frames', 'local_params']
        if attrs_names:
            self.attrs_names += attrs_names

        self.cur_step = 0
        self.traj_step = 1

    def __iter__(self):
        self.cur_step = 0
        for i in range(len(self)):

            yield self.cur_step
            self.cur_step += self.traj_step

        self.cur_step = 0

    origins = property(fset=lambda self, value: self._set_frame_attr('origins', value),
                       fget=lambda self: self._get_frame_attr('origins'))
    ref_frames = property(fset=lambda self, value: self._set_frame_attr('ref_frames', value),
                          fget=lambda self: self._get_frame_attr('ref_frames'))
    local_params = property(fset=lambda self, value: self._set_frame_attr('local_params', value),
                            fget=lambda self: self._get_frame_attr('local_params'))
    prot_origins = property(fset=lambda self, value: self._set_frame_attr('prot_origins', value),
                            fget=lambda self: self._get_frame_attr('prot_origins'))


class Tensor_Trajectory(Trajectory):
    @staticmethod
    def _is_plain_tensor_factory(traj_class):
        # Iterator passes torch.tensor; torch.tensor(zeros_tensor) triggers UserWarning.
        return traj_class is torch.tensor

    def _make_traj_storage(self, data):
        '''Plain torch.tensor factory: use zeros/slices directly; mod_Tensor gets (data, *attrs).'''
        if self._is_plain_tensor_factory(self.traj_class):
            if isinstance(data, torch.Tensor):
                return data.clone().detach()
            return torch.as_tensor(data, dtype=self.dtype)
        return self.traj_class(data, *self.traj_class_attrs)

    def __init__(self, dtype, traj_len, data_len, traj_class, *traj_class_attrs, attrs_names=None, shapes=None):
        self.shapes = [(traj_len, data_len, 1, 3), (traj_len, data_len, 3, 3), (traj_len, data_len, 6)]
        self.dtype = dtype
        self.data_len = data_len
        self.traj_class = traj_class
        self.traj_class_attrs = traj_class_attrs
        if shapes:
            self.shapes += shapes
        super().__init__(attrs_names)
        for shape, attr in zip(self.shapes, self.attrs_names):
            z = torch.zeros(*shape, dtype=dtype)
            setattr(self, f'{attr}_traj', z if self._is_plain_tensor_factory(traj_class) else traj_class(z, *traj_class_attrs))

    def copy(self, *traj_class_attrs):
        if not traj_class_attrs:
            traj_class_attrs = self.traj_class_attrs
        new = Tensor_Trajectory(bool, 1, 1, torch.tensor, attrs_names=self.attrs_names[3:], shapes=self.shapes[3:])
        for attr in self.attrs_names:
            setattr(new, f'{attr}_traj', self.traj_class(self.get_attr_trajectory(attr), *traj_class_attrs))

        return new

    def to(self, device):
        for attr in self.attrs_names:
            t = self.get_attr_trajectory(attr)
            if self._is_plain_tensor_factory(self.traj_class):
                setattr(self, f'{attr}_traj', t.to(device))
            else:
                setattr(self, f'{attr}_traj', self.traj_class(t, *self.traj_class_attrs).to(device))

    def extend(self, other_traj=None, **values_to_extend):

        for attr in self.attrs_names:
            tensor = self.get_attr_trajectory(attr)
            if other_traj:
                value = other_traj.get_attr_trajectory(attr)
            else:
                value = values_to_extend[attr]
            if isinstance(value, torch.Tensor):
                tail = value
            else:
                tail = torch.as_tensor(value, dtype=self.dtype, device=tensor.device)
            setattr(self, f'{attr}_traj', torch.concat([tensor, tail]))

    def add_attr(self, attr, shape):
        if attr in self.attrs:
            raise ValueError(f'Trajectory already has attribute {attr}')

        self.res_trajectory.attrs_names.append(attr)
        self.res_trajectory.shapes.append(shape)

        z = torch.zeros((len(self), *shape), dtype=self.dtype)
        setattr(
            self,
            f'{attr}_traj',
            z if self._is_plain_tensor_factory(self.traj_class) else self.traj_class(z, *self.traj_class_attrs),
        )

    def get_attr_trajectory(self, attr):
        return getattr(self, attr+'_traj')

    def _create_frame(self):
        for attr, shape in zip(self.attrs_names, self.shapes):
            attr += '_traj'
            tensor = self.get_attr_trajectory(attr)
            setattr(self, attr, torch.concat([tensor, torch.zeros(*shape[1:])]))

    def _get_frame_attr(self, attr, frame=None):
        if not frame:
            frame = self.cur_step
        return self.get_attr_trajectory(attr)[frame]

    def _set_frame_attr(self, attr, value, frame=None):
        if not frame:
            frame = self.cur_step
        if frame >= len(self):
            self._create_frame()
        self.get_attr_trajectory(attr)[frame] = value

    def __len__(self):
        return self.origins_traj.shape[0]

    def __getitem__(self, sl):

        traj_len, data_len = self.shapes[0][:2]
        new = Tensor_Trajectory(self.dtype, traj_len, data_len, self.traj_class, *self.traj_class_attrs)
        new.shapes = self.shapes
        new.attrs_names = self.attrs_names

        for attr in self.attrs_names:
            chunk = self.get_attr_trajectory(attr)[sl]
            setattr(new, f'{attr}_traj', new._make_traj_storage(chunk))

        return new


class H5_Trajectory(Trajectory):
    def __init__(self, filename, data_len, mode='r', attrs_names=None, shapes=None, string_format_val=5, **kwards):
        if shapes:
            self.shapes = shapes
        else:
            self.shapes = [(data_len, 1, 3), (data_len, 3, 3), (data_len, 6)]
        super().__init__(attrs_names)
        self.file = open_h5(filename, mode)
        self._dataset_kwards = kwards
        self.string_format_val = string_format_val
        if mode in ('w', 'x', 'w-'):
            self._last_frame_ind = -1

        elif mode == 'r':
            self._last_frame_ind = len(self.file) - 1

        elif mode in ('r+', 'a'):
            self._last_frame_ind = self.cur_step = len(self.file) - 1

    def _empty_dataset_kwargs(self):
        # h5py: creating dataset with only shape requires explicit dtype (default was float32 / 'f4').
        kw = dict(self._dataset_kwards)
        if 'dtype' not in kw and 'data' not in kw:
            kw['dtype'] = np.float32
        return kw

    def extend(self, other_traj=None, **values_to_extend):
        if other_traj:
            for step in other_traj:
                attrs = {}
                for attr in self.attrs_names:
                    attrs[attr] = other_traj._get_frame_attr(attr)
                self.add_frame(self._last_frame_ind+1, **attrs)

        else:
            for i in range(values_to_extend['origins_traj'].shape[0]):
                attrs = {}
                for attr in self.attrs_names:
                    attrs[attr] = values_to_extend[attr+'_traj']

                self.add_frame(self._last_frame_ind+1, **attrs)

    def add_attr(self, attr, shape):
        if attr in self.attrs_names:
            raise ValueError(f'Trajectory already has attribute {attr}')

        self.attrs_names.append(attr)
        self.shapes.append(shape)

        for i in range(len(self)):
            self.file[str(i).zfill(self.string_format_val)].create_dataset(
                attr, shape=shape, **self._empty_dataset_kwargs())

    def add_frame(self, step, **attrs):
        if step > self._last_frame_ind:
            self._create_frame(step)
        for attr, value in attrs.items():

            self.file[str(self._last_frame_ind).zfill(self.string_format_val)][attr][:] = value

    def copy(self, new):
        return self

    def _create_frame(self, frame_ind):
        if self._last_frame_ind < frame_ind:
            self._last_frame_ind = frame_ind

            frame_ind = str(frame_ind).zfill(self.string_format_val)
            frame = self.file.create_group(frame_ind)
            for attr_name, shape in zip(self.attrs_names, self.shapes):
                ds = frame.create_dataset(attr_name, shape=shape, **self._empty_dataset_kwargs())
        else:
            raise KeyError('frame creation failed, frame already exists')

    def _get_frame_attr(self, attr, frame=None):
        if not frame:
            frame = self.cur_step
        return torch.from_numpy(self.file[str(frame).zfill(self.string_format_val)][attr][:])

    def _set_frame_attr(self, attr, value, frame=None):
        if not frame:
            frame = self.cur_step
        if frame > self._last_frame_ind:
            self._create_frame(frame)
        self.file[str(frame).zfill(self.string_format_val)][attr][:] = value

    def __len__(self):
        return self._last_frame_ind + 1

    def get_attr_trajectory(self, attr, start=0, stop=None, step=1):
        attr_traj = np.zeros((self._last_frame_ind+1, *self.shapes[self.attrs_names.index(attr)]))
        if stop is None:
            stop = self._last_frame_ind + 1
        for i in range(start, stop, step):
            attr_traj[i] = self.file[str(i).zfill(self.string_format_val)][attr]

        return attr_traj

    def get_energy_array_slice(self, name, sl):
        if name != 'total':
            ind = ['bend', 'elst', 'ld', 'restr'].index(name)

        energy_arr = np.zeros((self._last_frame_ind))[sl]
        for en_i, i in enumerate(list(range(self._last_frame_ind))[sl]):
            if name == 'total':
                energy_arr[en_i] = np.sum(self.file[str(i).zfill(self.string_format_val)]['energies'][:])
            else:
                energy_arr[en_i] = self.file[str(i).zfill(self.string_format_val)]['energies'][ind]

        return energy_arr

    bend_energies = property(fget=lambda self: self.get_energies_arr(0))
    elst_energies = property(fget=lambda self: self.get_energies_arr(1))
    ld_energies = property(fget=lambda self: self.get_energies_arr(2))
    restr_energies = property(fget=lambda self: self.get_energies_arr(3))

    @property
    def total_energies(self):
        if not 'energies' in self.attrs_names:
            return None
        return self.bend_energies + self.elst_energies + self.ld_energies + self.restr_energies
