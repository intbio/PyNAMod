import unittest
from pynamod.geometry.geometrical_parameters import Geometrical_Parameters

class Test_Runner(unittest.TestCase):
    def test_frames_identity(self):
        nucl = DNA_structure_from_atomic(pdb_id='3LZ0',leading_strands=['I'],proteins=[])
        nucl.analyze_DNA()
        nucl.move_to_coord_center()

        old_geom_params = nucl.dna.geom_params

        new_params = Geometrical_Parameters(local_params = old_geom_params)

        ref_dif = old_geom_params.ref_frames - new_params.ref_frames
        assert ref_dif.abs().mean() < 10**-12
        ori_dif = old_geom_params.origins - new_params.origins
        assert ori_dif.abs().mean() < 10**-12
    
if __name__ == "__main__":
    unittest.main()
