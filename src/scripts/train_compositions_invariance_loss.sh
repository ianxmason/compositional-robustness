#!/bin/bash
#SBATCH -t 4:00:00                   #  walltime hh:mm:ss
#SBATCH -N 1                         #  one node
#SBATCH -n 18                        #  CPU cores
#SBATCH -x dgx001,dgx002,node[093,094,097,098,100,101,102,103,104,105,106]  #  had this issue https://github.mit.edu/MGHPCC/OpenMind/issues/3375
#SBATCH -o /om2/user/imason/compositions/slurm/EMNIST2/slurm-%j.out    # file to send output to
#SBATCH --array=0-14                 #  submit each of the 15 pairs of corruptions as a separate job
#SBATCH --mem=24G                    #  RAM
#SBATCH --gres=gpu:1                 #  one GPU
#SBATCH --constraint=11GB            #  any-gpu any gpu on cluster (may not be compatible with pytorch.
#                                    #  =high-capacity gives high-capacity GPU (compatible). =11GB gives 11gb gpu.

module load openmind/singularity/3.5.0
singularity exec --nv -B /om,/om2/user/$USER /om/user/xboix/singularity/xboix-tensorflow2.8.0.simg \
                python train_compositions_invariance_loss.py  --experiment-name "invariance-loss" \
                                                              --weights "0,0,0,10" \
                                                              --compare-corrs ",,,123" \
                                                              --pin-mem \
                                                              --check-if-run \
                                                              --corruption-ID $SLURM_ARRAY_TASK_ID

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
#                                                              --weights "10,10,0,10" \
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
#                python train_compositions_invariance_loss.py  --experiment-name "L1-L2-all-invariance" \
#                                                              --weights "10,10,0,0" \
#                                                              --compare-corrs "123,123,," \
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
