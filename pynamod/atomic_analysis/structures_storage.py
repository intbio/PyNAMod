import torch
import numpy as np

class Structures_Storage:
    def __init__(self,structure_class,structure_attrs_list,*stored_params):
        self.structure_class = structure_class
        self.structure_attrs_list = structure_attrs_list
        
        for name,value in zip(structure_attrs_list,stored_params):
            setattr(self,self.get_name(name),value)
            
        
    def append(self,*attrs):
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
                
        return self
            
    def _ls(self,item,attrs):
        return tuple(getattr(self,self.get_name(attr))[item] for attr in attrs)
            
    def _argsort(self,attrs):
        return sorted(range(len(getattr(self,self.get_name(attrs[0])))), key=lambda item: self._ls(item,attrs=attrs))
    
    
    def sort(self,*attrs):
        new_seq = self._argsort(attrs)
        
        for name in self.structure_attrs_list:
            sorted_data = [getattr(self,self.get_name(name))[i] for i in new_seq]
            if isinstance(sorted_data[0],torch.Tensor):
                sorted_data = torch.stack(sorted_data)
            setattr(self,self.get_name(name),sorted_data)
            
    def get_name(self,attr):
        return attr + 's' if not attr[-2:] == 'us' else attr[:-2] + 'i' 
            
            
    def __getitem__(self,sl):

        try:
            item_attrs = []
            for attr in self.structure_attrs_list:
                if isinstance(sl[0],bool):
                    item = [getattr(self,self.get_name(attr))[i] for i,bl in enumerate(sl) if bl]
                else:
                    item = [getattr(self,self.get_name(attr))[i] for i in sl]
                if isinstance(item[0],torch.Tensor):
                    item = torch.cat([i.reshape(1,*i.shape) for i in item])
                item_attrs.append(item)
                    
        except TypeError or IndexError:
            item_attrs = []
            for attr in self.structure_attrs_list:
                item = getattr(self,self.get_name(attr))
                if isinstance(item,torch.Tensor) and isinstance(sl,slice) and (sl.step and sl.step < 0):
                    item = torch.flip(item,(0,))
                    tens_sl = slice(sl.start,sl.stop,sl.step*-1)
                    item = item[tens_sl]
                else:
                    item = item[sl]
                item_attrs.append(item)
            
        if isinstance(sl,int) or isinstance(sl,np.int64):
            return self.structure_class(self,ind=sl)
        return type(self)(self.structure_class,self.structure_attrs_list,*item_attrs)
    
    def __add__(self,other):
        for attr in self.structure_attrs_list:
            self_attr = getattr(self,self.get_name(attr))
            other_attr = getattr(other,self.get_name(attr))

            if isinstance(self_attr,torch.Tensor):
                setattr(self,self.get_name(attr),torch.cat([self_attr,other_attr]))
            else:
                setattr(self,self.get_name(attr),self_attr+other_attr)
                        
                    
        return self
    
    def __len__(self):
        return len(getattr(self,self.structure_attrs_list[0]+'s'))
    
    
    def save_to_h5(self,file,group_name,**dataset_kwards):
        group = file.create_group(group_name)
        for attr in self.structure_attrs_list:
            attr = self.get_name(attr)
            if type(getattr(self,attr)[0]) in (int,str,bool,torch.Tensor,float):
                group.create_dataset(attr,data=getattr(self,attr),**dataset_kwards)
                
    def load_from_h5(self,file,group_name):
        for name,data in file[group_name].items():
            if len(data.shape) > 1:
                data = torch.tensor(data)
            else:
                data = list(data)
            
            
            setattr(self,name,data)
        for attr in self.structure_attrs_list:
            name = self.get_name(attr)
            if name not in file[group_name].keys():
                setattr(self,name,[None]*len(self))
                
    def copy(self):
        new = type(self)(self.structure_class,None)
        for attr in self.structure_attrs_list:
            if isinstance(getattr(self,self.get_name(attr)),torch.Tensor):
                setattr(new,self.get_name(attr),getattr(self,self.get_name(attr)).clone())
            else:
                setattr(new,self.get_name(attr),getattr(self,self.get_name(attr)).copy())
                
        return new
            
            
            
class Nucleotides_Storage(Structures_Storage):
    def __init__(self,nucleotide_class,u,*stored_params):
        self.mda_u = u
        structure_attrs_list = ['restype', 'resid', 'segid','leading_strand','ref_frame','origin','s_residue', 'e_residue','base_pair']
        if not stored_params:
            stored_params = [[],[],[],[],torch.empty(0,3,3,dtype=torch.double),torch.empty(0,1,3,dtype=torch.double),[],[],[]]
            
        super().__init__(nucleotide_class,structure_attrs_list,*stored_params)
        
    def copy(self):
        new = super().copy()
        new.mda_u = self.mda_u
        return new
    
    
class Pairs_Storage(Structures_Storage):
    def __init__(self,pair_class,nucleotides_storage,*stored_params):
        self.nucleotides_storage = nucleotides_storage
        structure_attrs_list = ['lead_nucl_ind', 'lag_nucl_ind', 'radius','charge','epsilon','geom_params']
        if not stored_params:
            stored_params = [[],[],[],[],[],[],[]]
            
        super().__init__(pair_class,structure_attrs_list,*stored_params)
        
    def _ls(self,item,attrs):
        nucl = self.nucleotides_storage[getattr(self,self.get_name(attrs[0]))[item]]
        return (nucl.leading_strand,nucl.resid)
    
    def sort(self):
        new_seq = self._argsort(['lead_nucl_ind'])
        
        for name in self.structure_attrs_list:
            sorted_data = [getattr(self,self.get_name(name))[i] for i in new_seq]
            if isinstance(sorted_data[0],torch.Tensor):
                sorted_data = torch.stack(sorted_data)
            setattr(self,self.get_name(name),sorted_data)
            
        leading_strands = self.nucleotides_storage[getattr(self,self.get_name('lead_nucl_ind'))].leading_strands
        if sum(leading_strands) == len(leading_strands):
            return self[leading_strands]
        else:
            return self[leading_strands] + self[[not i for i in leading_strands]]
        
    def __getitem__(self,sl):
        item = super().__getitem__(sl)
        if isinstance(item,Pairs_Storage):

            item.nucleotides_storage = self.nucleotides_storage
            return item

        else:
            return item
            
            
    def copy(self):
        new = super().copy()
        new.nucleotides_storage = self.nucleotides_storage
        return new