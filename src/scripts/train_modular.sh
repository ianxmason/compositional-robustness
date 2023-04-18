#!/bin/bash
#SBATCH -t 30:00:00                  #  walltime hh:mm:ss
#SBATCH -N 1                         #  one node
#SBATCH -n 4                         #  CPU cores
#SBATCH -x dgx001,dgx002,node[031,034,055,056,058,061,063,069,078,082,083,086,091,092,100-116]
#SBATCH -o /om2/user/imason/compositions/slurm/slurm-%j.out    # file to send output to
#SBATCH --array=0-5                  #  Modules/ImgSpace: 0-5. The elemental sets of corruptions as separate jobs
#SBATCH --mem=20G                    #  RAM.
#SBATCH --gres=gpu:1                 #  one GPU
#SBATCH --constraint=11GB            #

hostname
echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER
echo $SLURM_ARRAY_TASK_ID
module load openmind/singularity/3.5.0

### EMNIST ###
# Modules - Hardcoded Locations
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/imason/singularity/imason-pytorch.simg python train.py --dataset "EMNIST" --experiment "Modules" --total-n-classes 47 --max-epochs 200 --lr 1e-2 --weight 1 --corruption-ID $SLURM_ARRAY_TASK_ID --check-if-run
# Modules - Automatic Locations
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/imason/singularity/imason-pytorch.simg python train.py --dataset "EMNIST" --experiment "AutoModules" --total-n-classes 47 --max-epochs 200 --lr 1e-2 --weight 1 --corruption-ID $SLURM_ARRAY_TASK_ID --check-if-run
# ImgSpace - Train Autoencoders
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/imason/singularity/imason-pytorch.simg python train.py --dataset "EMNIST" --experiment "ImgSpace" --total-n-classes 47 --max-epochs 200 --lr 1e-2 --corruption-ID $SLURM_ARRAY_TASK_ID --n-workers 4 --check-if-run

### CIFAR ###
# Modules - Hardcoded Locations
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/imason/singularity/imason-pytorch.simg python train.py --dataset "CIFAR" --experiment "Modules" --total-n-classes 10 --max-epochs 200 --lr 1e-2 --weight 1 --corruption-ID $SLURM_ARRAY_TASK_ID --check-if-run
# Modules - Automatic Locations
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/imason/singularity/imason-pytorch.simg python train.py --dataset "CIFAR" --experiment "AutoModules" --total-n-classes 10 --max-epochs 200 --lr 1e-2 --weight 1 --corruption-ID $SLURM_ARRAY_TASK_ID --check-if-run
# ImgSpace - Train Autoencoders
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/imason/singularity/imason-pytorch.simg python train.py --dataset "CIFAR" --experiment "ImgSpace" --total-n-classes 10 --max-epochs 200 --lr 1e-2 --corruption-ID $SLURM_ARRAY_TASK_ID --n-workers 4 --check-if-run


### FACESCRUB ###
# Modules - Hardcoded Locations
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train.py --dataset "FACESCRUB" --experiment "Modules" --total-n-classes 388 --max-epochs 200 --lr 1e-2 --weight 1 --corruption-ID $SLURM_ARRAY_TASK_ID --check-if-run
# Modules - Automatic Locations
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train.py --dataset "FACESCRUB" --experiment "AutoModules" --total-n-classes 388 --max-epochs 200 --lr 1e-2 --weight 1 --corruption-ID $SLURM_ARRAY_TASK_ID --check-if-run
# ImgSpace - Train Autoencoders
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train.py --dataset "FACESCRUB" --experiment "ImgSpace" --total-n-classes 388 --max-epochs 200 --lr 1e-2 --corruption-ID $SLURM_ARRAY_TASK_ID --n-workers 4 --check-if-run
