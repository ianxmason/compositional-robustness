#!/bin/bash
#SBATCH -t 2:00:00                   #  walltime hh:mm:ss
#SBATCH -N 1                         #  one node
#SBATCH -n 2                         #  CPU cores
#SBATCH --mem=12G                    #  RAM
#SBATCH --array=0-6                  #  Generate the 7 elemental corruptions
#SBATCH -o <your_slurm_logging_path>/slurm-%j.out    # file to send output to

cd data/
#singularity exec --nv -B <your_bind_paths> <your_singularity_image_path> python emnist.py --corruption-ID $SLURM_ARRAY_TASK_ID
#singularity exec --nv -B <your_bind_paths> <your_singularity_image_path> python cifar.py --corruption-ID $SLURM_ARRAY_TASK_ID
#singularity exec --nv -B <your_bind_paths> <your_singularity_image_path> python facescrub.py --corruption-ID $SLURM_ARRAY_TASK_ID