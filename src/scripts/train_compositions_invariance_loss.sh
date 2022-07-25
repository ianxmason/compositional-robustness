#!/bin/bash
#SBATCH -t 6:00:00                  #  walltime hh:mm:ss
#SBATCH -N 1                         #  one node
#SBATCH -n 2                         #  CPU cores
#SBATCH -x dgx001,dgx002,node[093,094,097,098,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115]  #  had this issue https://github.mit.edu/MGHPCC/OpenMind/issues/3375
#SBATCH -o /om2/user/imason/compositions/slurm/EMNIST3/slurm-%j.out    # file to send output to
#SBATCH --array=0-14                 #  submit each of the 15 pairs of corruptions as a separate job. Or a hparam search.
#SBATCH --mem=12G                    #  RAM
#SBATCH --gres=gpu:1                 #  one GPU
#SBATCH --constraint=11GB            #  any-gpu any gpu on cluster (may not be compatible with pytorch.
#                                    #  =high-capacity gives high-capacity GPU (compatible). =11GB gives 11gb gpu.

hostname
echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER
echo $SLURM_ARRAY_TASK_ID

# Set time to 6:00:00
module load openmind/singularity/3.5.0
singularity exec --nv -B /om,/om2/user/$USER /om/user/xboix/singularity/xboix-tensorflow2.8.0.simg \
                python train_compositions_invariance_loss.py  --experiment-name "invariance-loss" \
                                                              --weights "0,0,0,1" \
                                                              --compare-corrs ",,,123" \
                                                              --pin-mem \
                                                              --check-if-run \
                                                              --corruption-ID $SLURM_ARRAY_TASK_ID

# Set time to 10:00:00
#module load openmind/singularity/3.5.0
#singularity exec --nv -B /om,/om2/user/$USER /om/user/xboix/singularity/xboix-tensorflow2.8.0.simg \
#                python train_compositions_invariance_loss.py  --experiment-name "L1-L2-fwd-invariance-loss" \
#                                                              --weights "1,1,0,1" \
#                                                              --compare-corrs "13,23,,123" \
#                                                              --pin-mem \
#                                                              --check-if-run \
#                                                              --corruption-ID $SLURM_ARRAY_TASK_ID

# Set time to 10:00:00
#module load openmind/singularity/3.5.0
#singularity exec --nv -B /om,/om2/user/$USER /om/user/xboix/singularity/xboix-tensorflow2.8.0.simg \
#                python train_compositions_invariance_loss.py  --experiment-name "L1-L2-bwd-invariance-loss" \
#                                                              --weights "1,1,0,1" \
#                                                              --compare-corrs "23,13,,123" \
#                                                              --pin-mem \
#                                                              --check-if-run \
#                                                              --corruption-ID $SLURM_ARRAY_TASK_ID

# Set time to 10:00:00
#module load openmind/singularity/3.5.0
#singularity exec --nv -B /om,/om2/user/$USER /om/user/xboix/singularity/xboix-tensorflow2.8.0.simg \
#                python train_compositions_invariance_loss.py  --experiment-name "L1-L2-all-invariance-loss" \
#                                                              --weights "1,1,0,1" \
#                                                              --compare-corrs "123,123,,123" \
#                                                              --pin-mem \
#                                                              --check-if-run \
#                                                              --corruption-ID $SLURM_ARRAY_TASK_ID











# Set time to 8:00:00
#module load openmind/singularity/3.5.0
#singularity exec --nv -B /om,/om2/user/$USER /om/user/xboix/singularity/xboix-tensorflow2.8.0.simg \
#                python train_compositions_invariance_loss.py  --experiment-name "supervised-invariance-loss" \
#                                                              --weights "0,0,0,10" \
#                                                              --compare-corrs ",,,123" \
#                                                              --pin-mem \
#                                                              --check-if-run \
#                                                              --corruption-ID $SLURM_ARRAY_TASK_ID

#module load openmind/singularity/3.5.0
#singularity exec --nv -B /om,/om2/user/$USER /om/user/xboix/singularity/xboix-tensorflow2.8.0.simg \
#                python train_compositions_invariance_loss.py  --experiment-name "L1-L2" \
#                                                              --weights "10,10,0,0" \
#                                                              --compare-corrs "12,13,," \
#                                                              --pin-mem \
#                                                              --check-if-run \
#                                                              --corruption-ID $SLURM_ARRAY_TASK_ID
##
#module load openmind/singularity/3.5.0
#singularity exec --nv -B /om,/om2/user/$USER /om/user/xboix/singularity/xboix-tensorflow2.8.0.simg \
#                python train_compositions_invariance_loss.py  --experiment-name "L1-L2-invariance-loss" \
#                                                              --weights "1,1,0,1" \
#                                                              --compare-corrs "12,13,,123" \
#                                                              --pin-mem \
#                                                              --check-if-run \
#                                                              --corruption-ID $SLURM_ARRAY_TASK_ID
#
#module load openmind/singularity/3.5.0
#singularity exec --nv -B /om,/om2/user/$USER /om/user/xboix/singularity/xboix-tensorflow2.8.0.simg \
#                python train_compositions_invariance_loss.py  --experiment-name "L2-L3-invariance-loss" \
#                                                              --weights "0,10,10,10" \
#                                                              --compare-corrs ",12,13,123" \
#                                                              --pin-mem \
#                                                              --check-if-run \
#                                                              --corruption-ID $SLURM_ARRAY_TASK_ID
#
#module load openmind/singularity/3.5.0
#singularity exec --nv -B /om,/om2/user/$USER /om/user/xboix/singularity/xboix-tensorflow2.8.0.simg \
#                python train_compositions_invariance_loss.py  --experiment-name "L1-L2-all-invariance-loss" \
#                                                              --weights "1,1,0,1" \
#                                                              --compare-corrs "123,123,,123" \
#                                                              --pin-mem \
#                                                              --check-if-run \
#                                                              --corruption-ID $SLURM_ARRAY_TASK_ID

#module load openmind/singularity/3.5.0
#singularity exec --nv -B /om,/om2/user/$USER /om/user/xboix/singularity/xboix-tensorflow2.8.0.simg \
#                python train_compositions_invariance_loss.py  --experiment-name "rotation-equivariant" \
#                                                              --weights "0,0,0,0" \
#                                                              --compare-corrs ",,," \
#                                                              --pin-mem \
#                                                              --check-if-run \
#                                                              --corruption-ID $SLURM_ARRAY_TASK_ID
#
#module load openmind/singularity/3.5.0
#singularity exec --nv -B /om,/om2/user/$USER /om/user/xboix/singularity/xboix-tensorflow2.8.0.simg \
#                python train_compositions_invariance_loss.py  --experiment-name "rotation-equivariant-invariance-loss" \
#                                                              --weights "0,0,0,10" \
#                                                              --compare-corrs ",,,123" \
#                                                              --pin-mem \
#                                                              --check-if-run \
#                                                              --corruption-ID $SLURM_ARRAY_TASK_ID
