# Graphical User Interface for Our Proposed Method
This package provides a graphical user interface (GUI) for the proposed method in the paper "Statistical Test for Feature Selection Pipelines by Selective Inference".

## Installation & Requirements
This package has the following dependencies:
- Python (version 3.10 or higher, we use 3.12.5)
    - numpy (version 1.26.4 or higher but lower than 2.0.0, we use 1.26.4)
    - scikit-learn (version 1.5.1 or higher, we use 1.5.1)
    - sicore (version 2.1.0 or higher, we use 2.1.0)
    - tqdm (version 4.66.5 or higher, we use 4.66.5)
    - streamlit (we use version 1.38.0, necessary for GUI)
    - barfi (we use version 0.7.0, necessary for GUI)

Please install these dependencies by pip.
```bash
pip install sicore # note that numpy is automatically installed by sicore
pip install scikit-learn
pip install tqdm
pip install streamlit # only for GUI
pip install barfi # only for GUI
```

## Usage
To run the demo, please execute the following command:
```
streamlit run si4pipeline.py
```
Then, you can access the demo by opening the browser and go to the following URL:
```http://localhost:8501```
