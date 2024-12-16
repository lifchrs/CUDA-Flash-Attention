#!/bin/bash
#SBATCH --account=m4776
#SBATCH -C gpu
#SBATCH --qos=shared
#SBATCH --time=00:10:00
#SBATCH -N 1
#SBATCH -n 1

srun --ntasks-per-node=1 dcgmi profile --pause
srun ncu -f -o reportx --target-processes all  ./build/attention 1 0 0 none none none none 4 4 4096 128 32 32 1 1
srun --ntasks-per-node=1 dcgmi profile --resume
