#!/bin/bash

orig_path=/home/jcmpos/dev
new_path="$orig_path"/code_copies/deepsplines_`date +%m-%d-%Y_%Hh%M`
mkdir -p "$new_path"

if [ "$3" = "hyperparam_tuning" ];
then
    hyper_folder="hyperparam_tuning"
    hyper_str="--""$3"
else
    hyper_folder="no_hyperparam_tuning"
    hyper_str=" "
fi

# create code copy with date/time stamp
log_dir="$orig_path"/experiments/cifar10/deeprelu
cp -r "$orig_path"/deepsplines "$new_path"

echo "BASH ARGS:"
echo "CUDA_VISIBLE_DEVICES=$1"
echo "taskset --cpu-list $2"

# try also setting bias=True in ResNet and removing
# weight decay on bias, weight inside basemodel.
for size in 33
do
    c_dir="$log_dir"/size"$size"/"$hyper_folder"
    mkdir -p "$c_dir"
    echo "$c_dir"
    eval CUDA_VISIBLE_DEVICES="$1" taskset --cpu-list "$2" python3 \
    "$orig_path"/deepsplines/scripts/gridsearch/torch_dataset/deepspline_torch_dataset.py \
    "$c_dir" cifar10 deeprelu --spline_size "$size" \
    --spline_range 4 --net resnet20 "$hyper_str" --aux_lr 1e-3
done
