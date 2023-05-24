#!/bin/bash
#SBATCH -t 30:00:00                  #  walltime hh:mm:ss
#SBATCH -N 1                         #  one node
#SBATCH -n 4                         #  CPU cores
#SBATCH -o <your_slurm_logging_path>/slurm-%j.out    # file to send output to
#SBATCH --array=0-5                  #  Modules/ImgSpace: 0-5. The elemental sets of corruptions as separate jobs
#SBATCH --mem=20G                    #  RAM.
#SBATCH --gres=gpu:1                 #  one GPU
#SBATCH --constraint=11GB            #

hostname
echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER
echo $SLURM_ARRAY_TASK_ID

### EMNIST ###
# Modules - Automatic Locations
#singularity exec --nv -B <your_bind_paths> <your_singularity_image_path> python train.py --dataset "EMNIST" --experiment "AutoModules" --total-n-classes 47 --max-epochs 200 --lr 1e-2 --weight 1.0 --corruption-ID $SLURM_ARRAY_TASK_ID --check-if-run
# ImgSpace - Train Autoencoders
#singularity exec --nv -B <your_bind_paths> <your_singularity_image_path> python train.py --dataset "EMNIST" --experiment "ImgSpace" --total-n-classes 47 --max-epochs 200 --lr 1e-1 --corruption-ID $SLURM_ARRAY_TASK_ID --n-workers 4 --check-if-run

### CIFAR ###
# Modules - Automatic Locations
#singularity exec --nv -B <your_bind_paths> <your_singularity_image_path> python train.py --dataset "CIFAR" --experiment "AutoModules" --total-n-classes 10 --max-epochs 200 --lr 1e-2 --weight 1.0 --corruption-ID $SLURM_ARRAY_TASK_ID --check-if-run
# ImgSpace - Train Autoencoders
#singularity exec --nv -B <your_bind_paths> <your_singularity_image_path> python train.py --dataset "CIFAR" --experiment "ImgSpace" --total-n-classes 10 --max-epochs 200 --lr 1e-1 --corruption-ID $SLURM_ARRAY_TASK_ID --n-workers 4 --check-if-run


### FACESCRUB ###
# Modules - Automatic Locations
#singularity exec --nv -B <your_bind_paths> <your_singularity_image_path> python train.py --dataset "FACESCRUB" --experiment "AutoModules" --total-n-classes 388 --max-epochs 200 --lr 1e-2 --weight 1e-1 --corruption-ID $SLURM_ARRAY_TASK_ID --check-if-run
# ImgSpace - Train Autoencoders
#singularity exec --nv -B <your_bind_paths> <your_singularity_image_path> python train.py --dataset "FACESCRUB" --experiment "ImgSpace" --total-n-classes 388 --max-epochs 200 --lr 1e-1 --corruption-ID $SLURM_ARRAY_TASK_ID --n-workers 4 --check-if-run
