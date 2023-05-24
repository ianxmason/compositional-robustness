#!/bin/bash
#SBATCH -t 0:30:00                   #  walltime hh:mm:ss.
#SBATCH -N 1                         #  one node
#SBATCH -n 8                         #  CPU cores
#SBATCH -o <your_slurm_logging_path>/slurm-%j.out    # file to send output to
#SBATCH --array=0-19                 #  split the ckpts to test into groups. 20 groups with 4 jobs in each.
#SBATCH --mem=16G                    #  RAM
#SBATCH --gres=gpu:1                 #  one GPU
#SBATCH --constraint=11GB            #

hostname
echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER
echo $SLURM_ARRAY_TASK_ID

jobs_per_gpu=4
for ((number=0; number<$jobs_per_gpu; number++))
do
  singularity exec --nv -B <your_bind_paths> <your_singularity_image_path> \
              python test.py --pin-mem \
                             --check-if-run \
                             --dataset "EMNIST" \
                             --total-n-classes 47 \
                             --experiment "CrossEntropy" \
                             --seed 38164641 \
                             --collect-activations \
                             --num-processes $(($SLURM_ARRAY_TASK_COUNT * $jobs_per_gpu)) \
                             --process $(($SLURM_ARRAY_TASK_ID * $jobs_per_gpu + $number)) &
done
wait