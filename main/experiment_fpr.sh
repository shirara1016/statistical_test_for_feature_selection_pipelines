# bash

for option in op1 op2 op12cv; do
    for seed in $(seq 0 9); do
        for n in 400 300 200 100; do
            python experiment/main_experiment.py \
                --seed $seed \
                --n $n \
                --option $option \
                >> result.txt 2>&1
        done
    done
done

for option in op1 op2 op12cv; do
    for seed in $(seq 0 9); do
        for n in 400 300 200 100; do
            python experiment/main_experiment.py \
                --seed $seed \
                --n $n \
                --option $option \
                --oc oc \
                >> result.txt 2>&1
        done
    done
done
