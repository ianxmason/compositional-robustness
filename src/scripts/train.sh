#!/bin/bash
#SBATCH -t 30:00:00                  #  walltime hh:mm:ss
#SBATCH -N 1                         #  one node
#SBATCH -n 4                         #  CPU cores
#SBATCH -o /om2/user/imason/compositions/slurm/EMNIST5/slurm-%j.out    # file to send output to
#SBATCH --array=0                    #  Modules/ImgSpace: 0-5. Contrastive/CE: 0. EMNIST5: 0-63. The elemental sets of corruptions as separate jobs
#SBATCH --mem=16G                    #  RAM.
#SBATCH --gres=gpu:1                 #  one GPU
#SBATCH --constraint=any-A100        #  any-gpu any gpu on cluster (may not be compatible with pytorch.
#                                    #  =high-capacity gives high-capacity GPU (compatible). =11GB gives 11gb gpu.

hostname
echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER
echo $SLURM_ARRAY_TASK_ID
module load openmind/singularity/3.5.0

# For modules. -t 12:00:00. --mem 12G. --array=0-5
# For contrastive/CE/img space. -t 48:00:00. --mem 24G. --array=0

# EMNIST
# Cross Entropy
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train.py --dataset "EMNIST" --experiment "CrossEntropy" --total-n-classes 47 --max-epochs 200 --lr 1e-2 --corruption-ID $SLURM_ARRAY_TASK_ID --n-workers 4 --check-if-run
# Contrastive loss
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train.py --dataset "EMNIST" --experiment "Contrastive" --total-n-classes 47 --max-epochs 200 --lr 1e-2 --weight 0.1 --corruption-ID $SLURM_ARRAY_TASK_ID --n-workers 4 --check-if-run
# Modules
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train.py --dataset "EMNIST" --experiment "Modules" --total-n-classes 47 --max-epochs 200 --lr 1e-2 --weight 1 --corruption-ID $SLURM_ARRAY_TASK_ID --check-if-run
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train.py --dataset "EMNIST" --experiment "AutoModules" --total-n-classes 47 --max-epochs 200 --lr 1e-2 --weight 1 --corruption-ID $SLURM_ARRAY_TASK_ID --check-if-run
# Autoencoders
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train.py --dataset "EMNIST" --experiment "ImgSpace" --total-n-classes 47 --max-epochs 200 --lr 1e-2 --corruption-ID $SLURM_ARRAY_TASK_ID --n-workers 4 --check-if-run

# CIFAR
# Cross Entropy
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train.py --dataset "CIFAR" --experiment "CrossEntropy" --total-n-classes 10 --max-epochs 200 --lr 1e-2 --corruption-ID $SLURM_ARRAY_TASK_ID --n-workers 4 --check-if-run
# Contrastive loss
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train.py --dataset "CIFAR" --experiment "Contrastive" --total-n-classes 10 --max-epochs 200 --lr 1e-2 --weight 0.1 --corruption-ID $SLURM_ARRAY_TASK_ID --n-workers 4 --check-if-run
# Modules
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train.py --dataset "CIFAR" --experiment "Modules" --total-n-classes 10 --max-epochs 200 --lr 1e-2 --weight 1 --corruption-ID $SLURM_ARRAY_TASK_ID --check-if-run
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train.py --dataset "CIFAR" --experiment "AutoModules" --total-n-classes 10 --max-epochs 200 --lr 1e-2 --weight 1 --corruption-ID $SLURM_ARRAY_TASK_ID --check-if-run
# Autoencoders
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train.py --dataset "CIFAR" --experiment "ImgSpace" --total-n-classes 10 --max-epochs 200 --lr 1e-2 --corruption-ID $SLURM_ARRAY_TASK_ID --n-workers 4 --check-if-run


# FACESCRUB
# Cross Entropy
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train.py --dataset "FACESCRUB" --experiment "CrossEntropy" --total-n-classes 388 --max-epochs 200 --lr 1e-2 --corruption-ID $SLURM_ARRAY_TASK_ID --n-workers 4 --check-if-run
# Contrastive loss
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train.py --dataset "FACESCRUB" --experiment "Contrastive" --total-n-classes 388 --max-epochs 200 --lr 1e-2 --weight 0.1 --corruption-ID $SLURM_ARRAY_TASK_ID --n-workers 4 --check-if-run
# Modules
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train.py --dataset "FACESCRUB" --experiment "Modules" --total-n-classes 388 --max-epochs 200 --lr 1e-2 --weight 1 --corruption-ID $SLURM_ARRAY_TASK_ID --check-if-run
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train.py --dataset "FACESCRUB" --experiment "AutoModules" --total-n-classes 388 --max-epochs 200 --lr 1e-2 --weight 1 --corruption-ID $SLURM_ARRAY_TASK_ID --check-if-run
# Autoencoders
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train.py --dataset "FACESCRUB" --experiment "ImgSpace" --total-n-classes 388 --max-epochs 200 --lr 1e-2 --corruption-ID $SLURM_ARRAY_TASK_ID --n-workers 4 --check-if-run


# CIFAR Modules
#python train.py --pin-mem --check-if-run --corruption-ID 0 --dataset "CIFAR" --total-n-classes 10 --max-epochs 200 --lr 1e-1 --experiment Modules --weights "1,1,1,1,1,1,1,1,1,1"

# FACESCRUB Modules
#python train.py --pin-mem --check-if-run --corruption-ID 0 --dataset "FACESCRUB" --total-n-classes 388 --max-epochs 200 --lr 1e-1 --experiment Modules --weights "1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"











# Below here is old code before changing optimizer, adding new datasets, refactoring.
# EMNIST will probably need to change lr and maybe max epochs


# Cross entropy loss -t 12:00:00
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train.py --corruption-ID $SLURM_ARRAY_TASK_ID --pin-mem --check-if-run
# Contrastive loss -t 14:00:00
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train.py --experiment "Contrastive" --weights "1,1,1,1,1,1" --corruption-ID $SLURM_ARRAY_TASK_ID --pin-mem --check-if-run
# Modules -t 14:00:00
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train.py --experiment "Modules" --weights "1,1,1,1,1,1" --corruption-ID $SLURM_ARRAY_TASK_ID --pin-mem --check-if-run

# V2 tests: Now includes automodules, autocontrastive and modlevel contrastive loss
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train.py --experiment "CrossEntropyV2" --corruption-ID $SLURM_ARRAY_TASK_ID --pin-mem --check-if-run
# Contrastive loss -t 14:00:00
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train.py --experiment "ContrastiveV2" --weights "1,1,1,1,1,1" --corruption-ID $SLURM_ARRAY_TASK_ID --pin-mem --check-if-run
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train.py --experiment "AutoContrastiveV2" --weights "1,1,1,1,1,1" --corruption-ID $SLURM_ARRAY_TASK_ID --pin-mem --check-if-run
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train.py --experiment "ModLevelContrastiveV2" --weights "1,1,1,1,1,1" --corruption-ID $SLURM_ARRAY_TASK_ID --pin-mem --check-if-run
# Modules -t 14:00:00
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train.py --experiment "ModulesV2" --weights "1,1,1,1,1,1" --corruption-ID $SLURM_ARRAY_TASK_ID --pin-mem --check-if-run
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train.py --experiment "AutoModulesV2" --weights "1,1,1,1,1,1" --corruption-ID $SLURM_ARRAY_TASK_ID --pin-mem --check-if-run
# Autoencoders -t 14:00:00
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train.py --experiment "ImgSpaceV2" --corruption-ID $SLURM_ARRAY_TASK_ID --pin-mem --check-if-run

# Contrastive Comparisons
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train.py --experiment "ModLevelContrastiveV3" --weights "1,1,1,1,1,1" --corruption-ID $SLURM_ARRAY_TASK_ID --pin-mem --check-if-run

#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train.py --experiment "ContrastiveL3W01" --weights "0.1,0.1,0.1,0.1,0.1,0.1" --corruption-ID $SLURM_ARRAY_TASK_ID --pin-mem --check-if-run
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train.py --experiment "ContrastiveL3W1" --weights "1,1,1,1,1,1" --corruption-ID $SLURM_ARRAY_TASK_ID --pin-mem --check-if-run
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train.py --experiment "ContrastiveL3W10" --weights "10,10,10,10,10,10" --corruption-ID $SLURM_ARRAY_TASK_ID --pin-mem --check-if-run

# Modules Comparisons
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train.py --experiment "ModulesV3" --weights "1,1,1,1,1,1" --corruption-ID $SLURM_ARRAY_TASK_ID --pin-mem --check-if-run
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train.py --experiment "ModulesV3NoPassThrough" --weights "1,1,1,1,1,1" --corruption-ID $SLURM_ARRAY_TASK_ID --pin-mem --check-if-run
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train.py --experiment "ModulesV3NoInvariance" --weights "0,0,0,0,0,0" --corruption-ID $SLURM_ARRAY_TASK_ID --pin-mem --check-if-run

#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train.py --experiment "AutoModulesV3" --weights "1,1,1,1,1,1" --corruption-ID $SLURM_ARRAY_TASK_ID --pin-mem --check-if-run
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train.py --experiment "AutoModulesV3NoPassThrough" --weights "1,1,1,1,1,1" --corruption-ID $SLURM_ARRAY_TASK_ID --pin-mem --check-if-run
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train.py --experiment "AutoModulesV3NoInvariance" --weights "0,0,0,0,0,0" --corruption-ID $SLURM_ARRAY_TASK_ID --pin-mem --check-if-run



