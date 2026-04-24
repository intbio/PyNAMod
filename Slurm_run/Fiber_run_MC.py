import pynamod
import argparse
import torch
import h5py

import warnings
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description='Start fiber simulations')

parser.add_argument('--h5file', action='store', dest='h5file',default=None,
                    help='cg structure file')
parser.add_argument('--target_accepted_steps','-tas', action='store', dest='target_accepted_steps',default=None,
                    help='target accepted steps')
parser.add_argument('--max_steps','-ms', action='store', dest='max_steps',default=None,
                    help='max_steps')
parser.add_argument('--traj_file', action='store', dest='traj_file',default=None,
                    help='traj file')

args = parser.parse_args()

cgs = pynamod.CG_Structure()
file = h5py.File(args.h5file,'r')
cgs.load_from_h5(file)
cgs.dna.transfer_trajectory_to_h5(args.traj_file,'w',compression="gzip",dtype=np.float16)

en = pynamod.Energy()
en.set_energy_matrices(cgs,ignore_neighbors=10)

intg = pynamod.Iterator(cgs,en,sigma_rot=0.15,sigma_transl=0.3)
intg.run(target_accepted_steps=int(args.target_accepted_steps),max_steps=int(args.max_steps),device='cuda',KT_factor=0.6,save_every=5,transfer_to_memory_every=50,dtype=torch.float,rebuild_every=400,mute=True)

cgs.dna.trajectory.file.close()
