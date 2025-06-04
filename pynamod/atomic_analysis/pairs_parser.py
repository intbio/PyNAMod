import numpy as np
import torch
import pickle
from itertools import combinations
from more_itertools import pairwise
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import pdist
from pynamod.geometry.geometrical_parameters import Geometrical_Parameters
from pynamod.atomic_analysis.structures_storage import Pairs_Storage

from importlib.resources import files

path = files('pynamod').joinpath('atomic_analysis/classifier.pkl')
with open(path, 'rb') as f:
    classifier = pickle.load(f)


class Base_Pair:
    def __init__(self, storage_class,lead_nucl_ind=None,
                 lag_nucl_ind=None,ind=None,lead_nucl=None,lag_nucl=None,radius=10,charge=-2,eps=0.5,geom_params=None):
        self.storage_class = storage_class
        self.ind = ind
        if lead_nucl:
            lead_nucl_ind = lead_nucl.ind
            lag_nucl_ind = lag_nucl.ind

        if ind is None:
            ind = len(storage_class)
            self.ind = ind
            storage_class.append(lead_nucl_ind,lag_nucl_ind,radius,charge,eps,geom_params)
            if storage_class.nucleotides_storage[lag_nucl_ind].leading_strand:
                print(1)
                self.lead_nucl_ind, self.lag_nucl_ind = self.lag_nucl_ind, self.lead_nucl_ind
        else:
            lead_nucl_ind = storage_class.lead_nucl_inds[ind]
            lag_nucl_ind = storage_class.lag_nucl_inds[ind]


        
        
        #self.pair_name = ''.join(sorted((lead_nucl.restype, lag_nucl.restype))).upper()
        

        
    def __lt__(self, other):
        return self.lead_nucl.__lt__(other.lead_nucl)
    
    def __repr__(self):
        return f'<Nucleotides pair with resids {self.lead_nucl.resid}, {self.lag_nucl.resid}, and segids {self.lead_nucl.segid}, {self.lag_nucl.segid}>'
    
    
    def get_pair_params(self):
        lead_nucl,lag_nucl = self.lead_nucl,self.lag_nucl
        ori = torch.vstack([lead_nucl.origin,lag_nucl.origin]).reshape(2,1,3)
        r_frames = torch.stack([lead_nucl.ref_frame,lag_nucl.ref_frame])
        
        if self.geom_params:
            self.geom_params.get_new_params_set(ref_frames=r_frames,origins=ori)
        else:
            self.geom_params = Geometrical_Parameters(ref_frames = r_frames, origins = ori,
                                                  pair_params=True)            

    
    def copy(self,**kwards):
        new = Base_Pair(lead_nucl = self.lead_nucl.copy(),lag_nucl = self.lag_nucl.copy(),radius=self.radius,charge=self.charge,eps=self.eps,dna_structure=self.dna_structure,geom_params=self.geom_params)
        new.update_references()
        for name,value in kwards.items():
            setattr(new,name,value)
        return new
    

    def get_params(self,attr):
        return getattr(self.geom_params,attr)[0]
    
    def set_params(self,value,attr):
        getattr(self.geom_params,attr)[0] = value
        
    def __setter(self,value,attr):
        getattr(self.storage_class,self.storage_class.get_name(attr))[self.ind] = value
        
    def __getter(self,attr):
        return getattr(self.storage_class,self.storage_class.get_name(attr))[self.ind]
        
    def __set_property(attr):
        setter = lambda self,value: self.__setter(value,attr=attr)
        getter = lambda self: self.__getter(attr=attr)
        return property(fset=setter,fget=getter)

    lead_nucl_ind = __set_property('lead_nucl_ind')
    lag_nucl_ind = __set_property('lag_nucl_ind')
    radius = __set_property('radius')
    charge = __set_property('charge')
    epsilon = __set_property('epsilon')
    geom_params = __set_property('geom_params')
    
        
    pair_params = property(fget=lambda self: self.get_params(attr='local_params'),fset=lambda self,value: self.set_params(attr='local_params'))
    Rm = property(fget=lambda self: self.get_params(attr='ref_frames'),fset=lambda self,value: self.set_params(attr='ref_frames'))
    om = property(fget=lambda self: self.get_params(attr='origins'),fset=lambda self,value: self.set_params(attr='origins'))
    
    @property
    def pair_name(self):
        if isinstance(self.storage_class.nucleotides_storage[int(self.lead_nucl_ind)].restype,bytes):
            return ''.join(sorted((str(self.storage_class.nucleotides_storage[int(self.lead_nucl_ind)].restype,"utf-8"), 
                               str(self.storage_class.nucleotides_storage[int(self.lag_nucl_ind)].restype,"utf-8")))).upper()
        else:
            return ''.join(sorted((str(self.storage_class.nucleotides_storage[int(self.lead_nucl_ind)].restype), 
                               str(self.storage_class.nucleotides_storage[int(self.lag_nucl_ind)].restype)))).upper()
    
    @property
    def lead_nucl(self):
        return self.storage_class.nucleotides_storage[self.lead_nucl_ind]
    
    @property
    def lag_nucl(self):
        return self.storage_class.nucleotides_storage[self.lag_nucl_ind]

def get_nucl_in_pairs_ind(nucleotides_storage):
    pair_names_check = [''.join(sorted((resname1, resname2))).upper() in ('AT', 'CG', 'AU')
                        for (resname1,resname2) in combinations(nucleotides_storage.restypes,2)]
    dist = pdist(nucleotides_storage.origins.reshape(-1,3))
    dist_check = dist < 4
    
    check = [name_check and d_check for name_check,d_check in zip(pair_names_check,dist_check)]
    
    rot_dif = [R.align_vectors(r1,r2)[0] for candidate,(r1,r2) in 
                zip(check,combinations(nucleotides_storage.ref_frames,2)) if candidate]
    rot_dif = R.concatenate(rot_dif).as_euler('zyx', degrees=True)
    rot_dif[:,2] += (rot_dif[:,2] < 0) * 360
    
    true_pairs = classifier.predict(np.hstack([rot_dif,dist[check].reshape(-1,1)])).astype(bool)
    
    nucl_ind = [(i1,i2) for candidate,(i1,i2) in zip(check,combinations(range(len(nucleotides_storage)),2)) if candidate]

    lead_ind,lag_ind = [],[]
    for true_pair,i in zip(true_pairs,nucl_ind):
        if true_pair:
            lead_ind.append(i[0])
            lag_ind.append(i[1])
    
    
    return lead_ind,lag_ind
    
    
def get_pairs(dna_structure,radius=10,charge=-2,eps=0.5):
    '''
    Get pairs of nucleotides for multiple structures based on RandomForest classifier algoritm
    '''
    nucleotides = dna_structure.nucleotides
    lead_ind,lag_ind = get_nucl_in_pairs_ind(nucleotides)
    
    ln = len(lead_ind)
    radii,charges,epsilons = [radius]*ln,[charge]*ln,[eps]*ln
    
    geom_params = []
    
    for ldi,lgi in zip(lead_ind,lag_ind):
        ori = torch.vstack([nucleotides.origins[ldi],nucleotides.origins[lgi]]).reshape(2,1,3)
        r_frames = torch.stack([nucleotides.ref_frames[ldi],nucleotides.ref_frames[lgi]])
        
        geom_params.append(Geometrical_Parameters(ref_frames = r_frames, origins = ori,
                                                  pair_params=True))

    pairs = Pairs_Storage(Base_Pair,nucleotides,lead_ind,lag_ind,radii,charges,epsilons,geom_params)

    
    return fix_missing_pairs(dna_structure,pairs)

    
def fix_missing_pairs(dna_structure,pairs):
    '''
Add missing pairs by context after classifier algorithm.
Fix the cases
A - T
G   C
T - A
by assuming the middle nucleotides are a pair.
-----
    '''
    nucleotides = dna_structure.nucleotides
    paired_nucl_ind = pairs.lead_nucl_inds + pairs.lag_nucl_inds

    for nucleotide in nucleotides[:-1]:

        if nucleotide.ind not in paired_nucl_ind and nucleotide.leading_strand:
            next_nucleotide = nucleotide.next_nucleotide
            
            prev_nucleotide = nucleotide.previous_nucleotide
    
            if prev_nucleotide.ind in paired_nucl_ind and next_nucleotide.ind in paired_nucl_ind:
                prev_pair_ind = pairs.lead_nucl_inds.index(prev_nucleotide.ind)
                next_pair_ind = pairs.lead_nucl_inds.index(next_nucleotide.ind)
                
                lag_ind1 = pairs.lag_nucl_inds[prev_pair_ind]
                lag_ind2 = pairs.lag_nucl_inds[next_pair_ind]
                
                pos_nucleotide1 = nucleotides[lag_ind1].previous_nucleotide
                pos_nucleotide2 = nucleotides[lag_ind2].next_nucleotide
                
                if pos_nucleotide1 and pos_nucleotide2:
                    
                    if pos_nucleotide1.ind == pos_nucleotide2.ind:
                        if ''.join(sorted((nucleotide.restype, pos_nucleotide1.restype))).upper() in ('AT', 'CG', 'AU'):
                            new_base_pair = Base_Pair(pairs,
                                                      lead_nucl=nucleotide, lag_nucl=pos_nucleotide1)
                            new_base_pair.get_pair_params()

    return pairs.sort()