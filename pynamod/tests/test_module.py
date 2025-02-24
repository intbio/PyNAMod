import pytest
#from pynamod.DNA.DNA_structure_analysis import DNA_structure_from_atomic
#from pynamod.geometry.bp_step_geometry import rebuild_by_full_par_frame_numba
import numpy as np
import warnings


def test_frames_identity():
    nucl = DNA_structure_from_atomic(pdb_id='3LZ0',leading_strands=['I'],proteins=[])
    nucl.analyze_DNA()
    nucl.move_to_coord_center()
    exp_ref_frames = rebuild_by_full_par_frame_numba(nucl.steps_params)
    dif = abs(exp_ref_frames - nucl.base_ref_frames)
    assert np.mean(dif) < 10**-12

