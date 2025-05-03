# Statistical Test for Feature Selection Pipelines by Selective Inference
This package is the implementation of the paper "Statistical Test for Feature Selection Pipelines by Selective Inference" for experiments.

## Installation & Requirements
This package has the following dependencies:
- Python (version 3.10 or higher, we use 3.12.5)
    - si4pipeline (version 1.0.1 or higher, we use 1.0.1)
    - numpy (version 1.26.4 or higher but lower than 2.0.0, we use 1.26.4)
    - scikit-learn (version 1.5.1 or higher, we use 1.5.1)
    - tqdm (version 4.66.5 or higher, we use 4.66.5)

Please install these dependencies by pip.
```bash
pip install si4pipeline # note that numpy is automatically installed by si4pipeline
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
