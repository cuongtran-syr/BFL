#!/usr/bin/env bash
d=$1
m=$2
s=$3
k=$4
t=$5
python3 run_privacy_exp.py --data $d --model_choice $m --sigma $s --K $k --seed $t  > result_${s}.out