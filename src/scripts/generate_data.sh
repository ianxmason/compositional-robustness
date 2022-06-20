#!/bin/bash
#SBATCH -t 24:00:00                  #  walltime hh:mm:ss
#SBATCH -N 1                         #  one node
#SBATCH -n 2                         #  CPU cores
#SBATCH --mem=24G                    #  RAM
#SBATCH -o /om2/user/imason/compositions/slurm/EMNIST2/slurm-%j.out    # file to send output to

# Took ?? hours and ??GB memory for initial long list of corruptions
cd data/
module load openmind/singularity/3.5.0
singularity exec --nv -B /om,/om2/user/$USER /om/user/xboix/singularity/xboix-tensorflow2.8.0.simg python emnist.py