#!/usr/bin/env bash
d=$1
m=$2
a=$3
b=$4
k=$5
t=$6
python3 run_non_private.py --data $d --model_choice $m --augment $a --bs $b --K $k --seed $t  > new_result_${t}.out