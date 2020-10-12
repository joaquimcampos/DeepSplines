#!/bin/bash

orig_path=/home/jcampos/repos
new_path="$orig_path"/code_copies/deepsplines_`date +%m-%d-%Y_%Hh%M`
mkdir -p "$new_path"

# create code copy with date/time stamp
log_dir="$orig_path"/experiments/cifar100/nin/apl
cp -f -r "$orig_path"/deepsplines "$new_path"

echo "BASH ARGS:"
echo "CUDA_VISIBLE_DEVICES=$1"
echo "taskset --cpu-list $2"

# try also setting bias=True in ResNet and removing
# weight decay on bias, weight inside basemodel.
for optimizer in 'mixed' 'SGD'
do
    c_dir="$log_dir""_""$optimizer"
    mkdir -p "$c_dir"
    echo "$c_dir"
    eval CUDA_VISIBLE_DEVICES="$1" taskset --cpu-list "$2" python3 \
    "$orig_path"/deepsplines/scripts/gridsearch/torch_dataset/apl_torch_dataset.py \
    "$c_dir" cifar100 \
    --optimizer "$optimizer" --net nin --weight_decay 1e-4
done
