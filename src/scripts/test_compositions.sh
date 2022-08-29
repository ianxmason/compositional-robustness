#!/bin/bash
#SBATCH -t 24:00:00                  #  walltime hh:mm:ss. EMNIST2: 2:00:00 or 2:30:00. EMNIST3 longer (many more testing corrs).
#SBATCH -N 1                         #  one node
#SBATCH -n 8                         #  CPU cores
#SBATCH -x dgx001,dgx002,node[093,094,097,098,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115]  #  had this issue https://github.mit.edu/MGHPCC/OpenMind/issues/3375
#SBATCH -o /om2/user/imason/compositions/slurm/EMNIST3/slurm-%j.out    # file to send output to
#SBATCH --array=0-56                 #  split the ckpts to test into groups. EMNIST2: 0-15. EMNIST3: 0-23
#SBATCH --mem=12G                    #  RAM
#SBATCH --gres=gpu:1                 #  one GPU 11gb
#SBATCH --constraint=11GB

hostname
echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER
echo $SLURM_ARRAY_TASK_ID

module load openmind/singularity/3.5.0

jobs_per_gpu=5
for ((number=0; number<$jobs_per_gpu; number++))
do
  singularity exec --nv -B /om,/om2/user/$USER /om/user/xboix/singularity/xboix-tensorflow2.9.simg \
              python test_compositions.py --pin-mem \
                                          --check-if-run \
                                          --num-processes $(($SLURM_ARRAY_TASK_COUNT * $jobs_per_gpu)) \
                                          --process $(($SLURM_ARRAY_TASK_ID * $jobs_per_gpu + $number)) &
done
wait
