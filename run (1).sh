#!/bin/bash -l
#SBATCH -t 2-00:00:00 # timelimit
#SBATCH -p gpu        # partition
#SBATCH -J CG_fibers_MC  # job name
#SBATCH -o log/ogmx.%j    # stdout filename %j will be replaced with job id
#SBATCH -e log/egmx.%j    # stderr filename
#SBATCH -N 1          # number of requested cluster nodes (servers)
#SBATCH --gres=gpu:1  # number of requested GPUs
#SBATCH --ntasks-per-node=2 # number of mpi ranks per node
#SBATCH --cpus-per-task=10  # number of cpu nodes per one MPI rank


conda init
conda activate pynamod
python Fiber_run_MC.py -tas 500000 -ms 3000000 --traj_file --h5file