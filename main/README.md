# Implementation of Our Proposed Method and Reproducibility of Experiments
This package is the implementation of the paper "Statistical Test for Feature Selection Pipelines by Selective Inference".
It provides (i) our implementation of the proposed method which is applicable to any pipeline configuration and (ii) the code for reproducing the results in the experiments section of the paper.

## Demonstration
The implementation we developed can be interactively executed using the provided `demonstration.ipynb` file.
This file contains a step-by-step guide on how to use the package and how to apply the proposed method to a given data analysis pipeline.

## Installation & Requirements
This package has the following dependencies:
- Python (version 3.10 or higher, we use 3.12.5)
    - numpy (version 1.26.4 or higher but lower than 2.0.0, we use 1.26.4)
    - scikit-learn (version 1.5.1 or higher, we use 1.5.1)
    - sicore (version 2.1.0 or higher, we use 2.1.0)
    - tqdm (version 4.66.5 or higher, we use 4.66.5)

Please install these dependencies by pip.
```bash
pip install sicore # note that numpy is automatically installed by sicore
pip install scikit-learn
pip install tqdm
```

## Reproducibility
To reproduce the results, please see the following instructions after installation step.
The results will be saved in "./results_*" folder as pickle file.
The plots will be saved in "./figures/main" folder as pdf file, which we have already got in advance.

For reproducing the figures in the left column of Figure 3 (type I error rate).
```bash
bash experiment_fpr.sh
```

For reproducing the figures in the right column of Figure 3 (power).
```bash
bash experiment_tpr.sh
```

For visualization of the reproduced results.
```bash
bash experiment_visualize.sh
```
