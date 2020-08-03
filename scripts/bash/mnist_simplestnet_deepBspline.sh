#!/bin/bash

orig_path=/home/jcmpos/dev
new_path="$orig_path"/code_copies/deepsplines_`date +%m-%d-%Y_%Hh%M`
mkdir -p "$new_path"

# create code copy with date/time stamp
log_dir="$orig_path"/experiments/mnist_multires/deepspline_gridsearch
cp -r "$orig_path"/deepsplines "$new_path"

echo "BASH ARGS:"
echo "CUDA_VISIBLE_DEVICES=$1"
echo "taskset --cpu-list $2"

for multires_milestones in "22 31 36" "14 22 26 31 36"
do
    for size in 3 5 11 21 41
    do
        c_dir="$log_dir"/size"$size"/"multires_"$(echo "$multires_milestones" | sed "s/ /_/g")
        mkdir -p "$c_dir"
        echo "$c_dir"
        eval CUDA_VISIBLE_DEVICES="$1" taskset --cpu-list "$2" python3 \
        "$orig_path"/deepsplines/scripts/gridsearch/torch_dataset/deepspline_torch_dataset.py \
        "$c_dir" mnist "$size" --lipschitz --hyperparam_tuning \
        --lr 1e-2 --aux_lr 1e-2 --spline_range 6 --multires_milestones "$multires_milestones"
    done
done
