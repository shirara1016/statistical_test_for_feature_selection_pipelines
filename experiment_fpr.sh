# bash

for option in op1 op2 all_cv; do
    for seed in $(seq 0 9); do
        for n in 400 300 200 100; do
            python experiment/experiment_main.py \
                --num_worker 4 \
                --option $option \
                --seed $seed \
                --n $n \
                >> result.txt 2>&1
        done
    done
done
