# bash

for option in op1 op2 all_cv; do
    for seed in $(seq 0 9); do
        for delta in 0.2 0.4 0.6 0.8; do
            python experiment/experiment_main.py \
                --num_worker 4 \
                --option $option \
                --seed $seed \
                --delta $delta \
                >> result.txt 2>&1
        done
    done
done
