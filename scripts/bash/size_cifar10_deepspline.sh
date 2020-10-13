#!/bin/bash

orig_path=/home/jcmpos/dev
new_path="$orig_path"/code_copies/deepsplines_`date +%m-%d-%Y_%Hh%M`
mkdir -p "$new_path"

# create code copy with date/time stamp
log_dir="$orig_path"/experiments/cifar10/resnet32/deepBspline_explicit_linear_bias_False_wd
cp -r "$orig_path"/deepsplines "$new_path"

echo "BASH ARGS:"
echo "CUDA_VISIBLE_DEVICES=$1"
echo "taskset --cpu-list $2"

# try also setting bias=True in ResNet and removing
# weight decay on bias, weight inside basemodel.
for size in 51 501 1001
do
    c_dir="$log_dir"/size"$size"
    mkdir -p "$c_dir"
    echo "$c_dir"
    eval CUDA_VISIBLE_DEVICES="$1" taskset --cpu-list "$2" python3 \
    "$orig_path"/deepsplines/scripts/gridsearch/torch_dataset/deepspline_torch_dataset.py \
    "$c_dir" cifar10 deepBspline_explicit_linear --spline_size "$size" \ 
    --spline_range 4 --net resnet32 --aux_lr 1e-3
done
