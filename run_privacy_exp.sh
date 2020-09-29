#!/usr/bin/env bash
d=$1
m=$2
s=$3
python3 run_privacy_exp.py --data $d --model_choice $m --seed $s  > result_${s}.out