#!/bin/bash
#SBATCH -t 14:00:00                  #  walltime hh:mm:ss
#SBATCH -N 1                         #  one node
#SBATCH -n 2                         #  CPU cores
#SBATCH -x dgx001,dgx002,node[093,094,097,098,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115]  #  had this issue https://github.mit.edu/MGHPCC/OpenMind/issues/3375
#SBATCH -o /om2/user/imason/compositions/slurm/EMNIST5/slurm-%j.out    # file to send output to
#SBATCH --array=0                  #  Modules: 0-5. Contrastive/CE: 0. EMNIST5: 0-63. The elemental sets of corruptions as separate jobs
#SBATCH --mem=12G                    #  RAM
#SBATCH --gres=gpu:1                 #  one GPU
#SBATCH --constraint=11GB            #  any-gpu any gpu on cluster (may not be compatible with pytorch.
#                                    #  =high-capacity gives high-capacity GPU (compatible). =11GB gives 11gb gpu.

hostname
echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER
echo $SLURM_ARRAY_TASK_ID
module load openmind/singularity/3.5.0

# Cross entropy loss -t 12:00:00
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train_emnist.py --corruption-ID $SLURM_ARRAY_TASK_ID --pin-mem --check-if-run
# Contrastive loss -t 14:00:00
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train_emnist.py --experiment "Contrastive" --weights "1,1,1,1,1,1" --corruption-ID $SLURM_ARRAY_TASK_ID --pin-mem --check-if-run
# Modules -t 14:00:00
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train_emnist.py --experiment "Modules" --weights "1,1,1,1,1,1" --corruption-ID $SLURM_ARRAY_TASK_ID --pin-mem --check-if-run

# V2 tests: Now includes automodules, autocontrastive and modlevel contrastive loss
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train_emnist.py --experiment "CrossEntropyV2" --corruption-ID $SLURM_ARRAY_TASK_ID --pin-mem --check-if-run
# Contrastive loss -t 14:00:00
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train_emnist.py --experiment "ContrastiveV2" --weights "1,1,1,1,1,1" --corruption-ID $SLURM_ARRAY_TASK_ID --pin-mem --check-if-run
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train_emnist.py --experiment "AutoContrastiveV2" --weights "1,1,1,1,1,1" --corruption-ID $SLURM_ARRAY_TASK_ID --pin-mem --check-if-run
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train_emnist.py --experiment "ModLevelContrastiveV2" --weights "1,1,1,1,1,1" --corruption-ID $SLURM_ARRAY_TASK_ID --pin-mem --check-if-run
# Modules -t 14:00:00
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train_emnist.py --experiment "ModulesV2" --weights "1,1,1,1,1,1" --corruption-ID $SLURM_ARRAY_TASK_ID --pin-mem --check-if-run
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train_emnist.py --experiment "AutoModulesV2" --weights "1,1,1,1,1,1" --corruption-ID $SLURM_ARRAY_TASK_ID --pin-mem --check-if-run


# Contrastive Comparisons
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train_emnist.py --experiment "ModLevelContrastiveV3" --weights "1,1,1,1,1,1" --corruption-ID $SLURM_ARRAY_TASK_ID --pin-mem --check-if-run

#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train_emnist.py --experiment "ContrastiveL3W01" --weights "0.1,0.1,0.1,0.1,0.1,0.1" --corruption-ID $SLURM_ARRAY_TASK_ID --pin-mem --check-if-run
#singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train_emnist.py --experiment "ContrastiveL3W1" --weights "1,1,1,1,1,1" --corruption-ID $SLURM_ARRAY_TASK_ID --pin-mem --check-if-run
singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train_emnist.py --experiment "ContrastiveL3W10" --weights "10,10,10,10,10,10" --corruption-ID $SLURM_ARRAY_TASK_ID --pin-mem --check-if-run

