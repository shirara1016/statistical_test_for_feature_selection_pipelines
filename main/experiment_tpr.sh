# bash

for option in op1 op2 op12cv; do
    for seed in $(seq 0 9); do
        for delta in 0.2 0.4 0.6 0.8; do
            python experiment/main_experiment.py \
                --seed $seed \
                --delta $delta \
                --option $option \
                >> result.txt 2>&1
        done
    done
done

for option in op1 op2 op12cv; do
    for seed in $(seq 0 9); do
        for delta in 0.2 0.4 0.6 0.8; do
            python experiment/main_experiment.py \
                --seed $seed \
                --delta $delta \
                --option $option \
                --oc oc \
                >> result.txt 2>&1
        done
    done
done
