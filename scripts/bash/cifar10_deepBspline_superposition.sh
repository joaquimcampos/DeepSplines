#!/bin/bash

orig_path=/home/jcmpos/dev
new_path="$orig_path"/code_copies/deepsplines_`date +%m-%d-%Y_%Hh%M`
mkdir -p "$new_path"

# create code copy with date/time stamp
log_dir="$orig_path"/experiments/cifar10/superposition_resnet20/deepspline_gridsearch
cp -r "$orig_path"/deepsplines "$new_path"

echo "BASH ARGS:"
echo "CUDA_VISIBLE_DEVICES=$1"
echo "taskset --cpu-list $2"

hyper_str=" "

for size in "3 17 33" "3 17 65" "3 33 129"
do
    c_dir="$log_dir"/"size_""$(echo "$size" | sed "s/ /_/g")"
    mkdir -p "$c_dir"
    echo "$c_dir"
    eval CUDA_VISIBLE_DEVICES="$1" taskset --cpu-list "$2" python3 \
    "$orig_path"/deepsplines/scripts/gridsearch/torch_dataset/deepspline_torch_dataset.py \
    "$c_dir" cifar10 deepBspline_superposition --spline_size "$size" \
    --spline_range 4 --net resnet20 "$hyper_str" --aux_lr 1e-3
done
