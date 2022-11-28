#!/bin/bash
#SBATCH -t 0:15:00                   #  walltime hh:mm:ss.
#SBATCH -N 1                         #  one node
#SBATCH -n 8                         #  CPU cores
#SBATCH -x dgx001,dgx002,node[093,094,097,098,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115]  #  had this issue https://github.mit.edu/MGHPCC/OpenMind/issues/3375
#SBATCH -o /om2/user/imason/compositions/slurm/EMNIST5/slurm-%j.out    # file to send output to
#SBATCH --array=0-19                 #  split the ckpts to test into groups. EMNIST 5: 0-15 with test-all off. 0-19 with test-all on. EMNIST 4: 0-15 with test-all off. 0-17 with test-all on.
#SBATCH --mem=16G                    #  RAM
#SBATCH --gres=gpu:1                 #  one GPU 11gb
#SBATCH --constraint=11GB

hostname
echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER
echo $SLURM_ARRAY_TASK_ID

module load openmind/singularity/3.5.0

# For V2 tests always leave --test-all on. Simply set --experiment to each of:
# "CrossEntropyV2"
# "ContrastiveV2"
# "AutoContrastiveV2"
# "ModLevelContrastiveV2"
# "ModulesV2"
# "AutoModulesV2"

# Trades off ram for runtime. Needs at least 16G, with 12G many jobs hang indefinitely.
jobs_per_gpu=4
for ((number=0; number<$jobs_per_gpu; number++))
do
  singularity exec --nv -B /om,/om2/user/$USER /om2/user/xboix/singularity/xboix-tensorflow2.9.simg \
              python test_emnist.py --pin-mem \
                                    --check-if-run \
                                    --test-all \
                                    --experiment "AutoModulesV2" \
                                    --num-processes $(($SLURM_ARRAY_TASK_COUNT * $jobs_per_gpu)) \
                                    --process $(($SLURM_ARRAY_TASK_ID * $jobs_per_gpu + $number)) &
done
wait