import unittest
import h5py
import numpy as np
import torch

import pynamod

from pynamod.atomic_analysis.pairs_parser import _OnnxPairClassifier


class Test_Runner(unittest.TestCase):
    def setUp(self):
        self.cgs = pynamod.CG_Structure(pdb_id='3lz0')

        self.cgs.analyze_dna(leading_strands=['I'])
        self.cgs.analyze_protein(n_cg_beads=80)
        self.cgs.dna.move_to_coord_center()

        self.correct_cgs = pynamod.CG_Structure()
        file = h5py.File('tests/cg_3lz0.h5', 'r')
        self.correct_cgs.load_from_h5(file)

        res1 = self.correct_cgs.dna.nucleotides[self.correct_cgs.dna.pairs.lead_nucl_inds].resids
        seg1 = self.correct_cgs.dna.nucleotides[self.correct_cgs.dna.pairs.lead_nucl_inds].segids
        res2 = self.correct_cgs.dna.nucleotides[self.correct_cgs.dna.pairs.lag_nucl_inds].resids
        seg2 = self.correct_cgs.dna.nucleotides[self.correct_cgs.dna.pairs.lag_nucl_inds].segids

        data = [[res1[i], seg1[i], res2[i], seg2[i]] for i in range(len(res1))]

        self.cgs_loaded_pairs = pynamod.CG_Structure(pdb_id='3lz0')

        self.cgs_loaded_pairs.analyze_dna(leading_strands=['I'],pairs_in_structure=data)
        self.cgs_loaded_pairs.analyze_protein(n_cg_beads=80)
        self.cgs_loaded_pairs.dna.move_to_coord_center()

    def test_nucleotides(self):

        assert len(self.cgs.dna.nucleotides) == len(self.correct_cgs.dna.nucleotides), 'wrong number of nucleotides in CG structure'
        assert len(self.cgs_loaded_pairs.dna.nucleotides) == len(self.correct_cgs.dna.nucleotides), 'wrong number of nucleotides in CG structure with loaded pairs'

        ref_dif = self.cgs.dna.nucleotides.ref_frames - self.correct_cgs.dna.nucleotides.ref_frames

        ref_dif = float(ref_dif.abs().mean())
        assert ref_dif < 10**-12, f'nucleotides ref frames difference: {ref_dif}'
        ori_dif = self.cgs.dna.nucleotides.origins - self.correct_cgs.dna.nucleotides.origins
        ori_dif = float(ori_dif.abs().mean())
        assert ori_dif < 10**-12, f'nucleotides origins difference: {ori_dif}'

    def test_pairs(self):

        assert len(self.cgs.dna.pairs) == len(self.correct_cgs.dna.pairs), f'wrong number of pairs in CG structure, ({len(self.cgs.dna.pairs)} instead of {len(self.correct_cgs.dna.pairs)})'
        assert len(self.cgs_loaded_pairs.dna.pairs) == len(self.correct_cgs.dna.pairs), f'wrong number of pairs in CG structure with loaded pairs, ({len(self.cgs_loaded_pairs.dna.pairs)} instead of {len(self.correct_cgs.dna.pairs)})'

        test_ref_frames = torch.vstack([p.geom_params.ref_frames[1].reshape(-1,3,3) for p in self.cgs.dna.pairs])
        correct_ref_frames = torch.vstack([p.geom_params.ref_frames[1].reshape(-1,3,3) for p in self.correct_cgs.dna.pairs])

        ref_dif = test_ref_frames - correct_ref_frames
        ref_dif = ref_dif.abs().mean()
        assert ref_dif < 10**-12, f'pairs ref frames difference: {ref_dif}'

        test_origins = torch.vstack([p.geom_params.origins[1].reshape(-1,3) for p in self.cgs.dna.pairs])
        correct_origins = torch.vstack([p.geom_params.origins[1].reshape(-1,3) for p in self.correct_cgs.dna.pairs])
        ori_dif = test_origins - correct_origins
        ori_dif = ori_dif.abs().mean()
        assert ori_dif < 10**-12, f'pairs origins difference: {ori_dif}'

    def test_params(self):
        ref_dif = self.cgs.dna.ref_frames - self.correct_cgs.dna.ref_frames
        ref_dif = ref_dif.abs().mean()
        assert ref_dif < 10**-12, f'step ref frames difference against correct structure: {ref_dif}'
        ori_dif = self.cgs.dna.origins - self.correct_cgs.dna.origins
        ori_dif = ori_dif.abs().mean()
        assert ori_dif < 10**-12, f'step origins difference against correct structure: {ori_dif}'        

    def test_geom_params(self):

        old_geom_params = self.cgs.dna.geom_params

        new_params = pynamod.geometry.geometrical_parameters.Geometrical_Parameters(local_params = old_geom_params.local_params)

        ref_dif = old_geom_params.ref_frames - new_params.ref_frames
        ref_dif = ref_dif.abs().mean()
        assert ref_dif < 10**-11, f'step ref frames after rebuild difference: {ref_dif}'
        ori_dif = old_geom_params.origins - new_params.origins
        ori_dif = ori_dif.abs().mean()
        assert ori_dif < 10**-11, f'step origins after rebuild difference: {ori_dif}'

    def test_MC_run(self):
        pass


class _FakeSession:
    def __init__(self, outputs):
        self.outputs = outputs

    def run(self, output_names, _inputs):
        return [self.outputs[name] for name in output_names]


class Test_OnnxPairClassifier(unittest.TestCase):
    def test_predict_prefers_probabilities_over_label(self):
        clf = _OnnxPairClassifier.__new__(_OnnxPairClassifier)
        setattr(clf, '_sess', _FakeSession({
            'probabilities': np.array([
                [-0.15, 0.15],
                [0.40, 0.60],
            ], dtype=np.float32),
            'label': np.array([1, 1], dtype=np.int64),
        }))
        clf._iname = 'float_input'
        clf._prob_name = 'probabilities'
        clf._label_name = 'label'

        pred = clf.predict(np.zeros((2, 4), dtype=np.float32))

        np.testing.assert_array_equal(pred, np.array([False, True]))

    def test_predict_falls_back_to_label_output(self):
        clf = _OnnxPairClassifier.__new__(_OnnxPairClassifier)
        setattr(clf, '_sess', _FakeSession({
            'label': np.array([0, 1], dtype=np.int64),
        }))
        clf._iname = 'float_input'
        clf._prob_name = None
        clf._label_name = 'label'

        pred = clf.predict(np.zeros((2, 4), dtype=np.float32))

        np.testing.assert_array_equal(pred, np.array([False, True]))


if __name__ == "__main__":
    unittest.main()
