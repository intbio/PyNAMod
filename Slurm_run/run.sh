#!/bin/bash -l
#SBATCH -t 2-00:00:00
#SBATCH -p gpu
#SBATCH -J CG_fibers_MC
#SBATCH -o log/slurm_out.%j
#SBATCH -e log/slurm_err.%j
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1

conda activate pynamod

python Fiber_run_MC.py -tas 1500000 -ms 6000000 --output_file --traj_file --h5file &
