#!/usr/bin/env bash
m=$1
s=$2
k=$3
t=$4
python3 run_covid_privacy.py  --model_choice $m --sigma $s --K $k --seed $t  > result_${s}.out