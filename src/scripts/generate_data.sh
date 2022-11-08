#!/bin/bash
#SBATCH -t 5:00:00                   #  walltime hh:mm:ss
#SBATCH -N 1                         #  one node
#SBATCH -n 2                         #  CPU cores
#SBATCH --mem=12G                    #  RAM
#SBATCH --array=0-148                #  EMNIST2: 57 corruptions. EMNIST3: 169 corrruptions. EMNIST4: 149 corruptions
#SBATCH -o /om2/user/imason/compositions/slurm/EMNIST4/slurm-%j.out    # file to send output to

# Took ~14-24 hours (not sure, stopped and restarted several times) and 4GB memory for initial long list of corruptions
cd data/
module load openmind/singularity/3.5.0
singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python emnist.py --corruption-ID $SLURM_ARRAY_TASK_ID