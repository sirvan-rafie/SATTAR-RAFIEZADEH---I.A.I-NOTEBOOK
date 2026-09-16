# AI7001 Credit Card Default Prediction

Practical Skills Assessment for **Introduction to Artificial Intelligence (AI7001)**  
Programme: **MSc in Artificial Intelligence**

## Project
Predictive Modelling Using Machine Learning - Credit Card Default Classification.

## Files
- `IAI_PRACTICAL_COMPLETE_ALL_5_TASKS.ipynb` - executed Jupyter Notebook covering all five assessment tasks.
- `Credit_Card.csv` - assessment dataset used by the notebook.
- `requirements.txt` - package versions used for the final reproducible run.

## Final methodology
The final notebook uses an 80:20 stratified train/test split. Model selection is performed with **3-fold cross-validation on the training data only**. Hyperparameter tuning uses `RandomizedSearchCV` with the **complete preprocessing + Random Forest pipeline inside every cross-validation fold**, so imputation, scaling and encoding are fitted separately within each fold.

The corrected tuning run selected:
- `n_estimators=80`
- `max_depth=10`
- `min_samples_leaf=2`
- `max_features='sqrt'`
- `class_weight='balanced'`
- Best CV F1: `0.5439`

Final tuned Random Forest test metrics:
- Accuracy: `0.7885`
- Precision: `0.5201`
- Recall: `0.5644`
- F1: `0.5414`
- AUC-ROC: `0.7707`

## Data-quality notes
- 4,788 exact duplicate rows are removed, leaving 30,000 records.
- `risk_leak` is excluded from modelling because it reveals target information.
- 291 rows have a `BILL_AMT_SUM` value that does not equal the row-wise sum of `BILL_AMT1` to `BILL_AMT6`; this is documented as a data-quality limitation.
- `RISK_RATING` is a deterministic grouping of `PAY_0`, so their feature-importance values should not be interpreted as independent evidence.

## How to run
1. Keep the notebook and `Credit_Card.csv` in the same folder.
2. Install the packages in `requirements.txt`.
3. Open the notebook in Jupyter Notebook, JupyterLab, VS Code or Google Colab.
4. Run the cells from top to bottom.

The notebook uses:
```python
DATA_PATH = "Credit_Card.csv"
```

## Assessment coverage
1. Exploratory Data Analysis
2. Data Preparation
3. Model Training and training-only model selection
4. Model Evaluation, Visualisation and leakage-safe Hyperparameter Tuning
5. Conclusion and Future Work

## Student
Sattar Rafiezadeh Naghani
