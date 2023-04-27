#!/bin/bash
#SBATCH -t 48:00:00                  #  walltime hh:mm:ss
#SBATCH -N 1                         #  one node
#SBATCH -n 4                         #  CPU cores
#SBATCH -x dgx001,dgx002,node[031,034,055,056,058,061,063,069,078,082,083,086,091,092,100-116]
#SBATCH -o /om2/user/imason/compositions/slurm/slurm-%j.out    # file to send output to
#SBATCH --array=0                    #  Contrastive/CE: 0.
#SBATCH --mem=24G                    #  RAM.
#SBATCH --gres=gpu:1                 #  one GPU
#SBATCH --constraint=11GB            #  another option is: any-A100

hostname
echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER
echo $SLURM_ARRAY_TASK_ID
module load openmind/singularity/3.5.0

### EMNIST ###
# Cross Entropy
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/imason/singularity/imason-pytorch.simg python train.py --dataset "EMNIST" --experiment "CrossEntropy" --total-n-classes 47 --max-epochs 200 --lr 1e-2 --corruption-ID $SLURM_ARRAY_TASK_ID --n-workers 4 --check-if-run
# Contrastive Loss
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/imason/singularity/imason-pytorch.simg python train.py --dataset "EMNIST" --experiment "Contrastive" --total-n-classes 47 --max-epochs 200 --lr 1e-2 --weight 1e-1 --corruption-ID $SLURM_ARRAY_TASK_ID --n-workers 4 --check-if-run
# ImgSpace - Train Classifiers
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/imason/singularity/imason-pytorch.simg python train.py --dataset "EMNIST" --experiment "ImgSpace" --total-n-classes 47 --max-epochs 200 --lr 1e-2 --corruption-ID $SLURM_ARRAY_TASK_ID --n-workers 4 --check-if-run

### CIFAR ###
# Cross Entropy
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/imason/singularity/imason-pytorch.simg python train.py --dataset "CIFAR" --experiment "CrossEntropy" --total-n-classes 10 --max-epochs 200 --lr 1e-1 --corruption-ID $SLURM_ARRAY_TASK_ID --n-workers 4 --check-if-run
# Contrastive Loss
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/imason/singularity/imason-pytorch.simg python train.py --dataset "CIFAR" --experiment "Contrastive" --total-n-classes 10 --max-epochs 200 --lr 1e-1 --weight 1e-1 --corruption-ID $SLURM_ARRAY_TASK_ID --n-workers 4 --check-if-run
# ImgSpace - Train Classifiers
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/imason/singularity/imason-pytorch.simg python train.py --dataset "CIFAR" --experiment "ImgSpace" --total-n-classes 10 --max-epochs 200 --lr 1e-1 --corruption-ID $SLURM_ARRAY_TASK_ID --n-workers 4 --check-if-run

### FACESCRUB ###
# Cross Entropy
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/imason/singularity/imason-pytorch.simg python train.py --dataset "FACESCRUB" --experiment "CrossEntropy" --total-n-classes 388 --max-epochs 200 --lr 1e-1 --corruption-ID $SLURM_ARRAY_TASK_ID --n-workers 4 --check-if-run
# Contrastive Loss
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/imason/singularity/imason-pytorch.simg python train.py --dataset "FACESCRUB" --experiment "Contrastive" --total-n-classes 388 --max-epochs 200 --lr 1e-1 --weight 1e-2 --corruption-ID $SLURM_ARRAY_TASK_ID --n-workers 4 --check-if-run
# ImgSpace - Train Classifiers
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/imason/singularity/imason-pytorch.simg python train.py --dataset "FACESCRUB" --experiment "ImgSpace" --total-n-classes 388 --max-epochs 200 --lr 1e-1 --corruption-ID $SLURM_ARRAY_TASK_ID --n-workers 4 --check-if-run
