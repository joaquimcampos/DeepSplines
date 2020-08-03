#!/bin/bash

orig_path=/home/jcmpos/dev
new_path="$orig_path"/code_copies/deepsplines_`date +%m-%d-%Y_%Hh%M`
mkdir -p "$new_path"

# create code copy with date/time stamp
log_dir="$orig_path"/experiments/additive_mnist_superposition/deepspline_gridsearch/aux_lr_"$3"
cp -r "$orig_path"/deepsplines "$new_path"

echo "BASH ARGS:"
echo "CUDA_VISIBLE_DEVICES=$1"
echo "taskset --cpu-list $2"

for size in "3 17 65 129" "3 17 65" "3 17 65 129 257"
do
    c_dir="$log_dir"/"size_""$(echo "$size" | sed "s/ /_/g")"
    mkdir -p "$c_dir"
    echo "$c_dir"
    eval CUDA_VISIBLE_DEVICES="$1" taskset --cpu-list "$2" python3 \
    "$orig_path"/deepsplines/scripts/gridsearch/torch_dataset/deepspline_torch_dataset.py \
    "$c_dir" mnist deepBspline_superposition --spline_size "$size" --lipschitz \
    --hyperparam_tuning --lr 1e-2 --aux_lr "$3" --spline_range 6
done
