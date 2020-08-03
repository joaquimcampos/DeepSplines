#!/bin/bash

orig_path=/home/jcmpos/dev
new_path="$orig_path"/code_copies
mkdir -p "$new_path"

# create code copy with date/time stamp
log_dir="$orig_path"/experiments/02_05_20_mnist_simplestnet/standard_gridsearch
cp -r "$orig_path"/deepsplines "$new_path"/deepsplines_`date +%m-%d-%Y_%Hh%M`

echo "BASH ARGS:"
echo "CUDA_VISIBLE_DEVICES=$1"
echo "taskset --cpu-list $2"

for lr in 1e-2 1e-3
do
    for activ in relu prelu
    do
        eval CUDA_VISIBLE_DEVICES="$1" taskset --cpu-list "$2" python3 \
        "$orig_path"/deepsplines/scripts/gridsearch/torch_dataset/standard_torch_dataset.py \
        "$log_dir"/"$activ" mnist "$activ" --lr "$lr"
    done
done
