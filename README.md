# EDA and Visualization Assignment II — Akhilesh S (23AM005)

## Dataset
Wisconsin Diagnostic Breast Cancer dataset loaded through `sklearn.datasets.load_breast_cancer`. The project uses the bundled public benchmark data (569 rows, 30 numerical features). Target encoding: 0 = malignant, 1 = benign.

## Environment
Python 3.10 or newer recommended.

## Install
```bash
pip install -r requirements.txt
```

## Run
```bash
python eda_project.py
```

The script creates `outputs/` containing nine figures and `metrics.json` / `metrics.txt`. It also writes `breast_cancer_dataset.csv`.

## Method
A stratified 70% training / 15% validation / 15% test split is used. Candidate MLP architectures and regularization values are compared on validation data. The selected pipeline is refit on the combined training and validation subsets, then evaluated once on the held-out test subset. Imputation and scaling occur inside the pipeline.

## Latest reproducible result (seed 42)
- Test accuracy: 0.9651 (96.51%)
- Test ROC-AUC (positive class = benign): 0.9942
- Confusion matrix in class order [malignant, benign]: `[[29, 3], [0, 54]]`

Metrics are split-specific and may vary with library versions. Educational project only; not for clinical diagnosis or treatment.
