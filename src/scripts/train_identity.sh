#!/bin/bash
#SBATCH -t 24:00:00                  #  walltime hh:mm:ss
#SBATCH -N 1                         #  one node
#SBATCH -n 4                         #  CPU cores
#SBATCH -x dgx001,dgx002,node[031,034,055,056,058,061,063,069,078,082,083,091,092]
#SBATCH -o /om2/user/imason/compositions/slurm/slurm-%j.out    # file to send output to
#SBATCH --array=0                    #
#SBATCH --mem=20G                    #  RAM.
#SBATCH --gres=gpu:1                 #  one GPU
#SBATCH --constraint=11GB            #

hostname
echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER
echo $SLURM_ARRAY_TASK_ID
module load openmind/singularity/3.5.0

### EMNIST ###
# Pretraining Network on Identity Data Before Adding Modules
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/imason/singularity/imason-pytorch.simg python train.py --dataset "EMNIST" --experiment "Modules" --total-n-classes 47 --max-epochs 200 --lr 1e-1 --weight 1 --corruption-ID $SLURM_ARRAY_TASK_ID --check-if-run

### CIFAR ###
# Pretraining Network on Identity Data Before Adding Modules
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/imason/singularity/imason-pytorch.simg python train.py --dataset "CIFAR" --experiment "Modules" --total-n-classes 10 --max-epochs 200 --lr 1e-1 --weight 1 --corruption-ID $SLURM_ARRAY_TASK_ID --check-if-run

### FACESCRUB ###
# Pretraining Network on Identity Data Before Adding Modules
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/imason/singularity/imason-pytorch.simg python train.py --dataset "FACESCRUB" --experiment "Modules" --total-n-classes 388 --max-epochs 200 --lr 1e-1 --weight 1 --corruption-ID $SLURM_ARRAY_TASK_ID --check-if-run
