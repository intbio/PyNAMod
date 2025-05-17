import warnings
warnings.filterwarnings('ignore')

import pynamod
import json
import torch
import h5py
import numpy as np
import time
from pynamod.geometry.trajectories import H5_Trajectory

file = h5py.File('NCP_with_tails_bluepr.h5','r')
results = {}
for i in range(5,31,5):

    nucl = pynamod.CG_Structure()
    
    nucl.load_from_h5(file)

    dna_gen = pynamod.CG_Structure()
    dna_gen.build_dna(sequence='atcg'*7)

    nucl.append_structures([dna_gen,nucl]*(i-1))

    en = pynamod.Energy(K_bend=1)
    en.set_energy_matrices(nucl,ignore_neighbors=10)


    traj = H5_Trajectory('test.h5',10000,len(nucl.dna.pairs_list),mode='w',compression="gzip",dtype=np.float16)
    intg = pynamod.Iterator(nucl,en,traj,sigma_rot=0.15,sigma_transl=0.3)
    movable = torch.ones(intg.dna_structure.dna.ref_frames.shape[0],dtype=bool)
    movable[0] = False
    start = 0
    for p in intg.dna_structure.proteins:
        stop = start + p.binded_dna_len
        movable[start:stop] = False
        start = stop + len(dna_gen.dna.pairs_list)
    movable[-p.binded_dna_len] = False
    t0 = time.time()
    intg.run(movable,target_accepted_steps=1e4,max_steps=1e8,device='cuda',KT_factor=0.6,
         save_every=10,transfer_to_memory_every=500,mute=True)
    dl = time.time() - t0
    results[i] = dl
    traj.file.close()
    
with open('results.json','w') as f:
    json.dump(results,f)