#!/bin/bash

orig_path=/home/jcampos/repos
new_path="$orig_path"/code_copies/deepsplines_`date +%m-%d-%Y_%Hh%M`
mkdir -p "$new_path"

# create code copy with date/time stamp
log_dir="$orig_path"/experiments/cifar100/"$3"/deepBspline_explicit_linear_bias_False_wd
cp -f -r "$orig_path"/deepsplines "$new_path"

echo "BASH ARGS:"
echo "CUDA_VISIBLE_DEVICES=$1"
echo "taskset --cpu-list $2"
echo "net $3"
echo "weight_decay $4"
echo "start_idx $5"
echo "end_idx $6"

# try also setting bias=True in ResNet and removing
# weight decay on bias, weight inside basemodel.
for size in 51 # 151 81 21 5
do
    c_dir="$log_dir"/size"$size"
    mkdir -p "$c_dir"
    echo "$c_dir"
    eval CUDA_VISIBLE_DEVICES="$1" taskset --cpu-list "$2" python3 \
    "$orig_path"/deepsplines/scripts/gridsearch/torch_dataset/deepspline_torch_dataset.py \
    "$c_dir" cifar100 deepBspline_explicit_linear --spline_size "$size" \
    --spline_range 4 --net "$3" --aux_lr 1e-3 --weight_decay "$4" \
    --start_idx "$5" --end_idx "$6"
done
