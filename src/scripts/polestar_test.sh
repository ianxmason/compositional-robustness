#!/bin/bash

# Runs on polestar from /compositions/src/ in tmux session where the following commmands have been run:
#module load openmind/singularity/3.5.0
#singularity shell --nv -B /om2/user/$USER /om2/user/imason/singularity/imason-pytorch.simg

# Then run e.g. ./scripts/polestar_test.sh -g 0 -t 2 -c 20
# Runs processes 8-11 of 80 on GPU 0

# Need to run this with t in 0-19 for each dataset and each experiment. Changing GPU as appropriate.
# And set --experiment to each of: "CrossEntropy" "Contrastive" "AutoModules" "ImgSpace"


while getopts g:t:c: flag   # g: gpu_idx, t: process to run c: total number of processes
do
    case "${flag}" in
        g) gpu=${OPTARG};;
        t) task_id=${OPTARG};;
        c) task_count=${OPTARG};;
    esac
done

jobs_per_gpu=4
for ((number=0; number<$jobs_per_gpu; number++))
do
  CUDA_VISIBLE_DEVICES=$gpu python test.py --pin-mem \
                                           --check-if-run \
                                           --dataset "CIFAR" \
                                           --total-n-classes 10 \
                                           --experiment "CrossEntropy" \
                                           --seed 24681012 \
                                           --collect-activations \
                                           --num-processes $(($task_count * $jobs_per_gpu)) \
                                           --process $(($task_id * $jobs_per_gpu + $number)) &
done
wait