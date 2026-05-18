import numpy as np
import torch
from MDAnalysis.core.groups import AtomGroup

from pynamod.geometry.geometrical_parameters import Geometrical_Parameters

'''
This module contains class Structures_Storage and its subclasses to store nucleotides and pairs data.
'''


class Structures_Storage:
    '''
    The purpose of this class is to mimic behaviour of a list of objects (currently Nucleotide or Base_Pair objects) by returning a proper obejct when index is used, but at the same type collect data from these objects in list-likes (lists, numpy arrays or torch Tensors) for an easier access to and analysis of their attributes. This class defines basic operations such as apppending, sorting and slicing.
    '''

    def __init__(self, structure_class, structure_attrs_list, *stored_params):

        # Will be used to return a single object in slicing
        self.structure_class = structure_class

        # List of all stored parameters, should be in singular form, plural will generated.
        self.structure_attrs_list = structure_attrs_list

        for name, value in zip(structure_attrs_list, stored_params):
            setattr(self, self.get_name(name), value)

<<<<<<< HEAD
    def append(self,*attrs):
        '''
        Appends given data to this object. All provided attributes should have the same length for a proper work of a class. Addition is defined to combine two objects.
        '''
        if len(attrs) == 1 and isinstance(attrs[0],self.structure_class):
            attrs = [getattr(attrs[0],name) for name in self.structure_attrs_list]

        for name,value in zip(self.structure_attrs_list,attrs):
            if isinstance(value,torch.Tensor):
                tens = getattr(self,self.get_name(name))
                if tens.dim() != value.dim():
                    value = value.reshape(1,*value.shape)
                setattr(self,self.get_name(name),torch.cat([tens,value]))

            else:
                getattr(self,self.get_name(name)).append(value)
=======
    def append(self, *attrs):
        '''
        Appends given data to this object. All provided attributes should have the same length for a proper work of a class. Addition is defined to combine two objects.
        '''
        if len(attrs) == 1 and isinstance(attrs[0], self.structure_class):
            attrs = [getattr(attrs[0], name) for name in self.structure_attrs_list]

        for name, value in zip(self.structure_attrs_list, attrs):
            if isinstance(value, torch.Tensor):
                tens = getattr(self, self.get_name(name))
                if tens.dim() != value.dim():
                    value = value.reshape(1, *value.shape)
                setattr(self, self.get_name(name), torch.cat([tens, value]))

            else:
                getattr(self, self.get_name(name)).append(value)
>>>>>>> 4c2722a3fcadace9ef261cd86d95e432a14f8376

        return self

    def _ls(self, item, attrs):
<<<<<<< HEAD
        return tuple(getattr(self,self.get_name(attr))[item] for attr in attrs)

    def _argsort(self, attrs):
        return sorted(range(len(getattr(self,self.get_name(attrs[0])))), key=lambda item: self._ls(item,attrs=attrs))
=======
        return tuple(getattr(self, self.get_name(attr))[item] for attr in attrs)

    def _argsort(self, attrs):
        return sorted(range(len(getattr(self, self.get_name(attrs[0])))), key=lambda item: self._ls(item, attrs=attrs))
>>>>>>> 4c2722a3fcadace9ef261cd86d95e432a14f8376

    def sort(self, *attrs):
        '''
        Sorts by given attributes.
        '''
        if len(self) == 0:
            return self
        new_seq = self._argsort(attrs)

        for name in self.structure_attrs_list:
<<<<<<< HEAD
            sorted_data = [getattr(self,self.get_name(name))[i] for i in new_seq]
            if isinstance(sorted_data[0],torch.Tensor):
                sorted_data = torch.stack(sorted_data)
            setattr(self,self.get_name(name),sorted_data)

    def get_name(self, attr):
        if attr[-2:] == 'us':
            return attr[:-2] + 'i'
        elif attr[-1] == 's':
            return attr
        else:
            return attr + 's'

    def __sl_isintance(self,sl_value,classes):
        return isinstance(sl_value,classes) or np.issubdtype(type(sl_value),classes[1]) or (isinstance(sl_value,torch.Tensor) and isinstance(sl_value.item(),classes))

    def __getitem__(self,sl):
=======
            attr_value = getattr(self, self.get_name(name))
            if isinstance(attr_value, torch.Tensor):
                sorted_data = attr_value[new_seq]
            else:
                sorted_data = [attr_value[i] for i in new_seq]
            setattr(self, self.get_name(name), sorted_data)
        return self

    def get_name(self, attr):
        return attr + 's' if not attr[-2:] == 'us' else attr[:-2] + 'i'

    def __sl_isintance(self, sl_value, classes):
        return isinstance(sl_value, classes) or np.issubdtype(type(sl_value), classes[1]) or (isinstance(sl_value, torch.Tensor) and isinstance(sl_value.item(), classes))

    def __getitem__(self, sl):
>>>>>>> 4c2722a3fcadace9ef261cd86d95e432a14f8376
        '''
        If sl is integer, returns instance of structure_class. Otherwise, returns the object of the same class as self. Slices support slice objects, list of indexes or boolean.
        '''

        if isinstance(sl, (slice, list, np.ndarray, torch.Tensor)):
            item_attrs = []
<<<<<<< HEAD
=======
            if isinstance(sl, torch.Tensor):
                sl = sl.detach().cpu().tolist()
            elif isinstance(sl, np.ndarray):
                sl = sl.tolist()
>>>>>>> 4c2722a3fcadace9ef261cd86d95e432a14f8376
            if isinstance(sl, slice):

                for attr in self.structure_attrs_list:
                    item = getattr(self, self.get_name(attr))[sl]
                    item_attrs.append(item)

<<<<<<< HEAD
            elif self.__sl_isintance(sl[0],(bool, np.bool_)):

                for attr in self.structure_attrs_list:
                    attr_val = getattr(self,self.get_name(attr))

                    if isinstance(attr_val,torch.Tensor):
                        item = attr_val[sl]
                    else:
                        item = [attr_val[i] for i,bl in enumerate(sl) if bl]

                    item_attrs.append(item)

            elif self.__sl_isintance(sl[0], (int, np.integer)):

                for attr in self.structure_attrs_list:
                    attr_val = getattr(self, self.get_name(attr))
=======
            elif len(sl) == 0:

                for attr in self.structure_attrs_list:
                    attr_val = getattr(self, self.get_name(attr))

                    if isinstance(attr_val, torch.Tensor):
                        item = attr_val[:0]
                    else:
                        item = []

                    item_attrs.append(item)

            elif self.__sl_isintance(sl[0], (bool, np.bool_)):
                sl = [bool(item) for item in sl]

                for attr in self.structure_attrs_list:
                    attr_val = getattr(self, self.get_name(attr))

                    if isinstance(attr_val, torch.Tensor):
                        item = attr_val[torch.tensor(sl, dtype=torch.bool)]
                    else:
                        item = [attr_val[i] for i, bl in enumerate(sl) if bl]

                    item_attrs.append(item)

            elif self.__sl_isintance(sl[0], (int, np.integer)):
                sl = [int(item) for item in sl]

                for attr in self.structure_attrs_list:
                    attr_val = getattr(self, self.get_name(attr))
>>>>>>> 4c2722a3fcadace9ef261cd86d95e432a14f8376
                    if isinstance(attr_val, torch.Tensor):
                        item = attr_val[sl]
                    else:
                        item = [attr_val[i] for i in sl]
                    item_attrs.append(item)

<<<<<<< HEAD
            if len(item_attrs[0]) == 1:
                if len(sl) == 1:
                    return self.structure_class(self, ind=sl[0])
                elif sum(sl) == 1:
                    return self.structure_class(self, ind=np.argwhere(sl)[0][0])

            return type(self)(self.structure_class,self.structure_attrs_list,*item_attrs)  

        elif isinstance(sl, int) or np.issubdtype(type(sl),np.integer):
            return self.structure_class(self,ind=sl)
=======
            new_storage = self.copy()
            for attr, item in zip(self.structure_attrs_list, item_attrs):
                setattr(new_storage, self.get_name(attr), item)
            return new_storage

        elif isinstance(sl, int) or np.issubdtype(type(sl), np.integer):
            return self.structure_class(self, ind=sl)
>>>>>>> 4c2722a3fcadace9ef261cd86d95e432a14f8376

        else:
            raise IndexError(f'Expected slice of type slice, list, np.ndarray, torch.Tensor or int, got {type(sl)}')

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

<<<<<<< HEAD
    def __add__(self,other):
        if not isinstance(other,(type(self),self.structure_class)):
            raise TypeError(f"TypeError: unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'")
        for attr in self.structure_attrs_list:

            self_attr = getattr(self,self.get_name(attr))
            if isinstance(other,type(self)):
                other_attr = getattr(other,self.get_name(attr))

            elif isinstance(other,self.structure_class):
                other_attr = getattr(other,attr)
                if isinstance(self_attr,torch.Tensor):
                    other_attr = other_attr[None,:]
                else:
                    other_attr = [other_attr]

            if isinstance(self_attr,torch.Tensor):
                setattr(self,self.get_name(attr),torch.cat([self_attr,other_attr]))
            else:
                setattr(self,self.get_name(attr),self_attr+other_attr)
=======
    def __add__(self, other):
        if not isinstance(other, (type(self), self.structure_class)):
            raise TypeError(f"TypeError: unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'")
        for attr in self.structure_attrs_list:

            self_attr = getattr(self, self.get_name(attr))
            if isinstance(other, type(self)):
                other_attr = getattr(other, self.get_name(attr))

            elif isinstance(other, self.structure_class):
                other_attr = getattr(other, attr)
                if isinstance(self_attr, torch.Tensor):
                    other_attr = other_attr[None, :]
                else:
                    other_attr = [other_attr]

            if isinstance(self_attr, torch.Tensor):
                setattr(self, self.get_name(attr), torch.cat([self_attr, other_attr]))
            else:
                setattr(self, self.get_name(attr), self_attr+other_attr)
>>>>>>> 4c2722a3fcadace9ef261cd86d95e432a14f8376

        return self

    def __len__(self):
<<<<<<< HEAD
        return len(getattr(self,self.get_name(self.structure_attrs_list[0])))

    def save_to_h5(self,file,group_name,**dataset_kwards):
=======
        return len(getattr(self, self.get_name(self.structure_attrs_list[0])))

    def save_to_h5(self, file, group_name, **dataset_kwards):
>>>>>>> 4c2722a3fcadace9ef261cd86d95e432a14f8376
        '''
        Saves all data to h5 file.

        Attributes:

        **file** - h5py File object.

        **group_name** - a name of a group that will be created to store data.

        **dataset_kwards** - attributes that will be used by create_dataset method (in addition to data).
        '''
        group = file.create_group(group_name)
        for attr in self.structure_attrs_list:
            attr = self.get_name(attr)
<<<<<<< HEAD
            item = getattr(self,attr)[0]
            if item is not None and not isinstance(item,(Geometrical_Parameters,AtomGroup)):
                group.create_dataset(attr,data=getattr(self,attr),**dataset_kwards)

            elif isinstance(getattr(self,attr)[0],Geometrical_Parameters):
                group.create_dataset('pair_params',data=torch.vstack([g.local_params[1] for g in getattr(self,attr)]),**dataset_kwards)
                group.create_dataset('pair_ref_frames',data=torch.vstack([g.ref_frames[1].reshape(-1,3,3) for g in getattr(self,attr)]),**dataset_kwards)
                group.create_dataset('pair_origins',data=torch.vstack([g.origins[1].reshape(-1,3) for g in getattr(self,attr)]),**dataset_kwards)

    def load_from_h5(self,file,group_name):
=======
            if not isinstance(getattr(self, attr)[0], (Geometrical_Parameters, AtomGroup)):
                group.create_dataset(attr, data=getattr(self, attr), **dataset_kwards)

            elif isinstance(getattr(self, attr)[0], Geometrical_Parameters):
                group.create_dataset('pair_params', data=torch.vstack([g.local_params[1] for g in getattr(self, attr)]), **dataset_kwards)
                group.create_dataset('pair_ref_frames', data=torch.vstack([g.ref_frames[1].reshape(-1, 3, 3) for g in getattr(self, attr)]), **dataset_kwards)
                group.create_dataset('pair_origins', data=torch.vstack([g.origins[1].reshape(-1, 3) for g in getattr(self, attr)]), **dataset_kwards)

    def load_from_h5(self, file, group_name):
>>>>>>> 4c2722a3fcadace9ef261cd86d95e432a14f8376
        '''
        Loasds data from h5 file and sets attributes of self with it.

        Attributes:

        **file** - h5py File object.

        **group_name** - a name of a group to load data from.
        '''
        for name, data in file[group_name].items():
            if len(data.shape) > 1:
                data = torch.tensor(np.asarray(data))
            else:
                data = list(data)
            if isinstance(data[0], bytes):
                data = [d.decode() for d in data]

<<<<<<< HEAD
            setattr(self,name,data)
=======
            setattr(self, name, data)
>>>>>>> 4c2722a3fcadace9ef261cd86d95e432a14f8376

        for attr in self.structure_attrs_list:
            name = self.get_name(attr)
            if name not in file[group_name].keys():
<<<<<<< HEAD
                setattr(self,name,[None]*len(self))
=======
                setattr(self, name, [None]*len(self))
>>>>>>> 4c2722a3fcadace9ef261cd86d95e432a14f8376

    def copy(self):
        new = type(self)(self.structure_class, None)
        for attr in self.structure_attrs_list:
            if isinstance(getattr(self, self.get_name(attr)), torch.Tensor):
                setattr(new, self.get_name(attr), getattr(self, self.get_name(attr)).clone())
            else:
<<<<<<< HEAD
                setattr(new,self.get_name(attr),getattr(self,self.get_name(attr)).copy())
=======
                setattr(new, self.get_name(attr), getattr(self, self.get_name(attr)).copy())
>>>>>>> 4c2722a3fcadace9ef261cd86d95e432a14f8376

        return new


class Nucleotides_Storage(Structures_Storage):
    '''
    Subclass that sets attributes to store nucleotides data.
    '''

    def __init__(self, nucleotide_class, u, *stored_params):
        self.mda_u = u
        structure_attrs_list = ['restype', 'resid', 'segid', 'leading_strand', 'ref_frame', 'origin', 's_residue', 'e_residue']
        if not stored_params:
<<<<<<< HEAD
            stored_params = [[],[],[],[],[],torch.empty(0,3,3,dtype=torch.double),torch.empty(0,1,3,dtype=torch.double),[],[]]

        super().__init__(nucleotide_class,structure_attrs_list,*stored_params)
=======
            stored_params = [[], [], [], [], torch.empty(0, 3, 3, dtype=torch.double), torch.empty(0, 1, 3, dtype=torch.double), [], []]

        super().__init__(nucleotide_class, structure_attrs_list, *stored_params)
>>>>>>> 4c2722a3fcadace9ef261cd86d95e432a14f8376

    def copy(self):
        new = super().copy()
        new.mda_u = self.mda_u
        return new

    def __repr__(self):
        return f'<Storage with {len(self)} nucleotides>'


class Pairs_Storage(Structures_Storage):
    '''
    Subclass that sets attributes to store pairs data.
    '''

    def __init__(self, pair_class, nucleotides_storage, *stored_params):
        self.nucleotides_storage = nucleotides_storage
        structure_attrs_list = ['lead_nucl_ind', 'lag_nucl_ind', 'radius', 'charge', 'epsilon', 'geom_params']
        if not stored_params:
<<<<<<< HEAD
            stored_params = [[],[],[],[],[],[],[]]

        super().__init__(pair_class,structure_attrs_list,*stored_params)

    def _ls(self,item,attrs):
        nucl = self.nucleotides_storage[getattr(self,self.get_name(attrs[0]))[item]]
        return (nucl.leading_strand,nucl.resid)
=======
            stored_params = [[], [], [], [], [], [], []]

        super().__init__(pair_class, structure_attrs_list, *stored_params)

    def _ls(self, item, attrs):
        nucl = self.nucleotides_storage[getattr(self, self.get_name(attrs[0]))[item]]
        return (nucl.leading_strand, nucl.resid)
>>>>>>> 4c2722a3fcadace9ef261cd86d95e432a14f8376

    def sort(self):
        if len(self) == 0:
            return self.copy()
        new_seq = self._argsort(['lead_nucl_ind'])

        for name in self.structure_attrs_list:
<<<<<<< HEAD
            sorted_data = [getattr(self,self.get_name(name))[i] for i in new_seq]
            if isinstance(sorted_data[0],torch.Tensor):
                sorted_data = torch.stack(sorted_data)
            setattr(self,self.get_name(name),sorted_data)

        leading_strands = self.nucleotides_storage[getattr(self,self.get_name('lead_nucl_ind'))].leading_strands
=======
            attr_value = getattr(self, self.get_name(name))
            if isinstance(attr_value, torch.Tensor):
                sorted_data = attr_value[new_seq]
            else:
                sorted_data = [attr_value[i] for i in new_seq]
            setattr(self, self.get_name(name), sorted_data)

        leading_strands = self.nucleotides_storage[getattr(self, self.get_name('lead_nucl_ind'))].leading_strands
>>>>>>> 4c2722a3fcadace9ef261cd86d95e432a14f8376
        if sum(leading_strands) == len(leading_strands):
            result = self[leading_strands]
            if not isinstance(result, Pairs_Storage):
                # Create a new Pairs_Storage with the single item
                new_storage = Pairs_Storage(self.structure_class, self.nucleotides_storage)
                new_storage.append(result)
                return new_storage
            return result
        else:
            return self[leading_strands] + self[[not i for i in leading_strands]]

<<<<<<< HEAD
    def __getitem__(self,sl):
=======
    def __getitem__(self, sl):
>>>>>>> 4c2722a3fcadace9ef261cd86d95e432a14f8376
        item = super().__getitem__(sl)
        if isinstance(item, Pairs_Storage):
            item.nucleotides_storage = self.nucleotides_storage
            return item

        else:
            return item

    def __repr__(self):
        return f'<Storage with {len(self)} pairs>'

    def copy(self):
        new = super().copy()
        new.nucleotides_storage = self.nucleotides_storage
        return new

<<<<<<< HEAD
    def load_from_h5(self,file,group_name):
        super().load_from_h5(file,group_name)
        if 'pair_params' in file[group_name].keys():
            pair_params = torch.tensor(file[group_name]['pair_params'])
            ref_frames = torch.tensor(file[group_name]['pair_ref_frames'])
            origins = torch.tensor(file[group_name]['pair_origins'])
            params_sets = [[torch.vstack([torch.zeros(1,*p.shape),p.reshape(1,*p.shape)]) for p in (pars,oris,rfs)] for (pars,rfs,oris) in zip(pair_params,ref_frames,origins)]
            self.geom_params = [Geometrical_Parameters(local_params=pars,origins=oris,ref_frames=rfs) for (pars,oris,rfs) in params_sets]
=======
    def load_from_h5(self, file, group_name):
        super().load_from_h5(file, group_name)

        pair_params = torch.as_tensor(np.asarray(file[group_name]['pair_params']))
        ref_frames = torch.as_tensor(np.asarray(file[group_name]['pair_ref_frames']))
        origins = torch.as_tensor(np.asarray(file[group_name]['pair_origins']))
        params_sets = [[torch.vstack([torch.zeros(1, *p.shape), p.reshape(1, *p.shape)]) for p in (pars, oris, rfs)] for (pars, rfs, oris) in zip(pair_params, ref_frames, origins)]
        self.geom_paramss = [Geometrical_Parameters(local_params=pars, origins=oris, ref_frames=rfs) for (pars, oris, rfs) in params_sets]
>>>>>>> 4c2722a3fcadace9ef261cd86d95e432a14f8376
