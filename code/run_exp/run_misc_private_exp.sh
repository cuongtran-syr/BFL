#!/usr/bin/env bash
d=$1
m=$2
s=$3
k=$4
t=$5
python3 run_incorrect_files2.py --data $d --model_choice $m --sigma $s --K $k --seed $t  > new_result_${s}.out