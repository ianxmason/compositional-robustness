#!/bin/bash
#SBATCH -t 2:00:00                   #  walltime hh:mm:ss
#SBATCH -N 1                         #  one node
#SBATCH -n 2                         #  CPU cores
#SBATCH --mem=12G                    #  RAM
#SBATCH --array=0-6                  #  Generate the 7 elemental corruptions
#SBATCH -o /om2/user/imason/compositions/slurm/slurm-%j.out    # file to send output to

cd data/
module load openmind/singularity/3.5.0
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/imason/singularity/imason-pytorch.simg python emnist.py --corruption-ID $SLURM_ARRAY_TASK_ID
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/imason/singularity/imason-pytorch.simg python cifar.py --corruption-ID $SLURM_ARRAY_TASK_ID
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/imason/singularity/imason-pytorch.simg python facescrub.py --corruption-ID $SLURM_ARRAY_TASK_ID