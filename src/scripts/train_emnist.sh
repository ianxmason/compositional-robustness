#!/bin/bash
#SBATCH -t 12:00:00                  #  walltime hh:mm:ss
#SBATCH -N 1                         #  one node
#SBATCH -n 2                         #  CPU cores
#SBATCH -x dgx001,dgx002,node[093,094,097,098,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115]  #  had this issue https://github.mit.edu/MGHPCC/OpenMind/issues/3375
#SBATCH -o /om2/user/imason/compositions/slurm/EMNIST4/slurm-%j.out    # file to send output to
#SBATCH --array=0-127                #  EMNIST4: 0-127. The elemental sets of corruptions as separate jobs
#SBATCH --mem=12G                    #  RAM
#SBATCH --gres=gpu:1                 #  one GPU
#SBATCH --constraint=11GB            #  any-gpu any gpu on cluster (may not be compatible with pytorch.
#                                    #  =high-capacity gives high-capacity GPU (compatible). =11GB gives 11gb gpu.

hostname
echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER
echo $SLURM_ARRAY_TASK_ID

module load openmind/singularity/3.5.0
singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg python train_emnist.py --corruption-ID $SLURM_ARRAY_TASK_ID --pin-mem --check-if-run