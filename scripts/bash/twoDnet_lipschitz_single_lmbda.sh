#!/bin/bash

code_path=/home/jcampos/repos/deepsplines
log_dir="$code_path"/experiments/lipschitz_twoDnet/lmbda_1e-4

for size in 3 5 7 11 15 21 31 51 81 101 121 151
do
    eval python3 "$code_path"/scripts/gridsearch/twoD/deepspline_twoD.py \
        "$log_dir" circle_1500 twoDnet_onehidden "$size" \
        --start_idx 4 --end_idx 5 --lipschitz --hidden 2
done
