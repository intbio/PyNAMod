
from importlib.resources import files
from itertools import combinations

import numpy as np
import onnxruntime as ort
import torch
from more_itertools import pairwise
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation as R

from pynamod.atomic_analysis.structures_storage import Pairs_Storage
from pynamod.geometry.geometrical_parameters import Geometrical_Parameters

'''
This module determines all base pairs of nucleotides in a structure. The analysis is performed by get_pairs and fix_missing_pairs functions, and the representation of the data is given by Base_Pair class similarly to Nucleotide class. The analysis starts by detecting all possible pair candidates that are Watson–Crick type and and the distance between nucleotides is less than 4 Å. True pairs are than chosen with RandomForest classifier, the trained model is saved in classifier.onnx. Missing pairs are restored by their context. Classifier can mistakenly assign nucleotide into two or more pairs, which was adressed in model training, which focused on decreasing the number of false positive pairs.
'''


class _OnnxPairClassifier:
    '''Wraps ONNX Runtime inference for the pair RandomForest model.'''

    def __init__(self, model_bytes: bytes):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self._sess = ort.InferenceSession(
            model_bytes, sess_options=so, providers=['CPUExecutionProvider']
        )
        self._iname = self._sess.get_inputs()[0].name
        onames = [o.name for o in self._sess.get_outputs()]
        label_name = next((n for n in onames if 'label' in n.lower()), onames[0])
        self._oname = label_name

    def predict(self, X):
        x = np.asarray(X, dtype=np.float32)
        y = self._sess.run([self._oname], {self._iname: x})[0]
        if y.dtype.kind == 'f':
            return (y > 0.5).astype(bool).reshape(-1)
        return y.astype(bool).reshape(-1)


_onnx_path = files('pynamod').joinpath('atomic_analysis/classifier.onnx')
classifier = _OnnxPairClassifier(_onnx_path.read_bytes())


class Base_Pair:
    '''
    Represents data of a single base pair. It exists as a reference to a pairs storage object which contains data of all pairs in a structure. All of its main
    variables are properties that get information from the storage class or set it there. These variables include:

    **lead_nucl**, **lag_nucl** - Nucleotide objects.

    **lead_nucl_ind**, **lag_nucl_ind** - indices to locate pair nucleotides in nucleotide storage.

    **pair_name** - string, consists of sorted types of Nucleotides in this pair.

    **radius**, **charge**, **epsilon** - parameters of a pair for energy functions.

    **geom_params** - Geometrical_Parameters object that stores local geometrical parameters of a pair as well as reference middle frame and origin of a pair. Note that this is different from step parameters, as it calculates parameters between two nucleotides.

    **pair_params**, **Rm**, **om** - torch Tensors of pair geometrical parameters, reference middle frame and origin.

    '''

    def __init__(self, storage_class, lead_nucl_ind=None,
                 lag_nucl_ind=None, ind=None, lead_nucl=None, lag_nucl=None, radius=10, charge=-2, eps=0.5, geom_params=None):
        self.storage_class = storage_class
        self.ind = ind

        if lead_nucl:
            lead_nucl_ind = lead_nucl.ind
            lag_nucl_ind = lag_nucl.ind

        if ind is None:
            ind = len(storage_class)
            self.ind = ind
            storage_class.append(lead_nucl_ind, lag_nucl_ind, radius, charge, eps, geom_params)

            if not storage_class.nucleotides_storage[lead_nucl_ind].leading_strand:
                self.lead_nucl_ind, self.lag_nucl_ind = self.lag_nucl_ind, self.lead_nucl_ind

        else:
            lead_nucl_ind = storage_class.lead_nucl_inds[ind]
            lag_nucl_ind = storage_class.lag_nucl_inds[ind]

    def __lt__(self, other):
        return self.lead_nucl.__lt__(other.lead_nucl)

    def __eq__(self, other):
        return self.storage_class == other.storage_class and self.ind == other.ind

    def __repr__(self):
        return f'<Nucleotides pair with resids {self.lead_nucl.resid}, {self.lag_nucl.resid}, and segids {self.lead_nucl.segid}, {self.lag_nucl.segid}>'

    def get_pair_params(self):
        '''
        Sets geometrical parameters of the pair. If geometrical parameters are already stored new ones will be added as a new frame. Otherwise, Geometrical_Parameters object is created.
        '''
        lead_nucl, lag_nucl = self.lead_nucl, self.lag_nucl
        ori = torch.vstack([lead_nucl.origin, lag_nucl.origin]).reshape(2, 1, 3)
        r_frames = torch.stack([lead_nucl.ref_frame, lag_nucl.ref_frame])

        if self.geom_params:
            self.geom_params.get_new_params_set(ref_frames=r_frames, origins=ori)
        else:
            self.geom_params = Geometrical_Parameters(ref_frames=r_frames, origins=ori,
                                                      pair_params=True)

    def copy(self, **kwards):
        new = Base_Pair(lead_nucl=self.lead_nucl.copy(), lag_nucl=self.lag_nucl.copy(), radius=self.radius,
                        charge=self.charge, eps=self.eps, dna_structure=self.dna_structure, geom_params=self.geom_params)
        new.update_references()
        for name, value in kwards.items():
            setattr(new, name, value)
        return new

    def set_params(self, value, sl, attr):
        getattr(self.geom_params, attr)[sl] = value

    def _setter(self, value, attr):
        getattr(self.storage_class, self.storage_class.get_name(attr))[self.ind] = value

    def _getter(self, attr):
        return getattr(self.storage_class, self.storage_class.get_name(attr))[self.ind]

    def _set_property(attr):
        def setter(self, value): return self._setter(value, attr=attr)
        def getter(self): return self._getter(attr=attr)
        return property(fset=setter, fget=getter)

    lead_nucl_ind = _set_property('lead_nucl_ind')
    lag_nucl_ind = _set_property('lag_nucl_ind')
    radius = _set_property('radius')
    charge = _set_property('charge')
    epsilon = _set_property('epsilon')
    geom_params = _set_property('geom_params')

    pair_params = property(fget=lambda self: self.geom_params.local_params[1], fset=lambda self, value: self.set_params(value, 1, attr='local_params'))
    Rm = property(fget=lambda self: self.geom_params.Rm[0], fset=lambda self, value: self.set_params(value, 0, attr='Rm'))
    om = property(fget=lambda self: self.geom_params.om[0], fset=lambda self, value: self.set_params(value, 0, attr='om'))

    @property
    def pair_name(self):
        if isinstance(self.lead_nucl.restype, bytes):
            return ''.join(sorted((str(self.lead_nucl.restype, "utf-8"),
                                   str(self.lag_nucl.restype, "utf-8")))).upper()
        else:
            return ''.join(sorted((str(self.lead_nucl.restype),
                                   str(self.lag_nucl.restype)))).upper()

    @property
    def lead_nucl(self):
        return self.storage_class.nucleotides_storage[self.lead_nucl_ind]

    @property
    def lag_nucl(self):
        return self.storage_class.nucleotides_storage[self.lag_nucl_ind]


def _get_nucl_in_pairs_ind(nucleotides_storage):
    '''
    Applies classifier to pairs candidates and returns indices of nucleotides in found pairs.
    '''

    pair_names_check = [''.join(sorted((resname1, resname2))).upper() in ('AT', 'CG', 'AU')
                        for (resname1, resname2) in combinations(nucleotides_storage.restypes, 2)]
    dist = pdist(nucleotides_storage.origins.reshape(-1, 3))
    dist_check = dist < 4

    check = [name_check and d_check for name_check, d_check in zip(pair_names_check, dist_check)]
    if not any(check):
        return [], []

    rot_dif = [R.align_vectors(r1, r2)[0] for candidate, (r1, r2) in
               zip(check, combinations(nucleotides_storage.ref_frames, 2)) if candidate]
    if len(rot_dif) == 0:
        return [], []
    rot_dif = R.concatenate(rot_dif).as_euler('zyx', degrees=True)
    rot_dif[:, 2] += (rot_dif[:, 2] < 0) * 360

    true_pairs = classifier.predict(np.hstack([rot_dif, dist[check].reshape(-1, 1)])).astype(bool)

    nucl_ind = [(i1, i2) for candidate, (i1, i2) in zip(check, combinations(range(len(nucleotides_storage)), 2)) if candidate]

    lead_ind, lag_ind = [], []
    for true_pair, i in zip(true_pairs, nucl_ind):
        if true_pair:
            lead_ind.append(i[0])
            lag_ind.append(i[1])

    return lead_ind, lag_ind


def _fix_missing_pairs(dna_structure, pairs):
    '''
    Adds missing pairs by context after classifier algorithm.
    Fixes the cases
    A - T
    G   C
    C - G
    by assuming the middle nucleotides are a pair.
    '''
    nucleotides = dna_structure.nucleotides
    paired_nucl_ind = pairs.lead_nucl_inds + pairs.lag_nucl_inds
    lead_to_pair = {lead_ind: pair_ind for pair_ind, lead_ind in enumerate(pairs.lead_nucl_inds)}

    for nucleotide in nucleotides[:-1]:

        if nucleotide.ind not in paired_nucl_ind and nucleotide.leading_strand:
            next_nucleotide = nucleotide.next_nucleotide

            prev_nucleotide = nucleotide.previous_nucleotide

            if next_nucleotide and prev_nucleotide and prev_nucleotide.ind in paired_nucl_ind and next_nucleotide.ind in paired_nucl_ind:
                # Some nucleotides can be paired only as lagging partners, so skip them instead of raising.
                if prev_nucleotide.ind not in lead_to_pair or next_nucleotide.ind not in lead_to_pair:
                    continue
                prev_pair_ind = lead_to_pair[prev_nucleotide.ind]
                next_pair_ind = lead_to_pair[next_nucleotide.ind]

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


def get_pairs(dna_structure, radius=10, charge=-2, eps=0.5):
    '''
    Runs the analysis to determine all pairs of nucleotides for a given structure.

    Attributes:

    **dna_structure** - DNA_Structure object to analyze.

    **radius**, **charge**, **eps** - values of parameters to set for each pair.

    Returns:

    Sorted Pairs_Storage objects.
    '''
    nucleotides = dna_structure.nucleotides
    lead_ind, lag_ind = _get_nucl_in_pairs_ind(nucleotides)

    ln = len(lead_ind)
    radii, charges, epsilons = [radius]*ln, [charge]*ln, [eps]*ln

    geom_params = []

    for ldi, lgi in zip(lead_ind, lag_ind):
        ori = torch.vstack([nucleotides.origins[ldi], nucleotides.origins[lgi]]).reshape(2, 1, 3)
        r_frames = torch.stack([nucleotides.ref_frames[ldi], nucleotides.ref_frames[lgi]])

        geom_params.append(Geometrical_Parameters(ref_frames=r_frames, origins=ori,
                                                  pair_params=True))

    pairs = Pairs_Storage(Base_Pair, nucleotides, lead_ind, lag_ind, radii, charges, epsilons, geom_params)
    pairs = _fix_missing_pairs(dna_structure, pairs)

    return pairs
