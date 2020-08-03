#!/bin/bash

orig_path=/home/jcmpos/dev
new_path="$orig_path"/code_copies/deepsplines_`date +%m-%d-%Y_%Hh%M`
mkdir -p "$new_path"

# create code copy with date/time stamp
log_dir="$orig_path"/experiments/cifar10/multires_resnet20/deepspline_gridsearch
cp -r "$orig_path"/deepsplines "$new_path"

echo "BASH ARGS:"
echo "CUDA_VISIBLE_DEVICES=$1"
echo "taskset --cpu-list $2"

hyper_str=" "

for size in 3 5
do
    for multires_milestones in "150 225 262" "50 100 150 200 250" "40 70 100 130 160 190 210"
    do
        c_dir="$log_dir"/size"$size"/"multires_""$(echo "$multires_milestones" | sed "s/ /_/g")"
        mkdir -p "$c_dir"
        echo "$c_dir"
        eval CUDA_VISIBLE_DEVICES="$1" taskset --cpu-list "$2" python3 \
        "$orig_path"/deepsplines/scripts/gridsearch/torch_dataset/deepspline_torch_dataset.py \
        "$c_dir" cifar10 deepBspline_explicit_linear --spline_size "$size" \
        --spline_range 4 --net resnet20 "$hyper_str" --aux_lr 1e-3\
        --multires_milestones "$multires_milestones"
    done
done
