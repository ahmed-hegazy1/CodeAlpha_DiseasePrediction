# Disease Prediction from Medical Data
**CodeAlpha Machine Learning Internship – Task 4**

A machine learning project that predicts whether a breast tumor is malignant or benign from diagnostic measurements, and compares four classification algorithms.

## Objective
Predict the possibility of disease from structured patient data using classification techniques, as described in the CodeAlpha Task 4 brief.

## Dataset
- **Source:** Breast Cancer Wisconsin (Diagnostic) dataset, loaded directly from `scikit-learn` (`load_breast_cancer`). Originally from the UCI ML Repository.
- **Size:** 569 samples, 30 numeric features (mean, error, and "worst" values of radius, texture, perimeter, area, smoothness, etc.) and 1 target.
- **Target:** `0` = malignant, `1` = benign.
- **Class Distribution:** 212 malignant, 357 benign.

## Approach
1. Load the data into a pandas DataFrame and inspect it.
2. Split into 80% training / 20% testing (`random_state=42`, `stratify=y` to keep the class ratio equal in both sets).
3. Scale features with `StandardScaler` (fitted on the training set only to avoid data leakage).
4. Train and compare four algorithms:
   - Logistic Regression
   - Support Vector Machine (RBF kernel)
   - Random Forest (100 trees)
   - XGBoost
5. Evaluate with accuracy, precision, recall, F1-score, and a confusion matrix.

## Results
Evaluated on the 114-sample test set:

| Model | Accuracy | Recall (malignant) | Precision (malignant) |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** | **98.25%** | **0.98** | **0.98** |
| **SVM (RBF)** | **98.25%** | **0.98** | **0.98** |
| **Random Forest** | 95.61% | 0.93 | 0.95 |
| **XGBoost** | 95.61% | 0.90 | 0.97 |

- **Confusion Matrix (Logistic Regression):** 41 malignant and 71 benign cases were classified correctly, with 1 false negative (malignant predicted as benign) and 1 false positive.

## Key Takeaways
- Linear models (Logistic Regression and SVM) performed slightly better than tree-based models on this dataset.
- In medical screening, **recall on the malignant class** matters most, since a false negative means a missed diagnosis. Logistic Regression and SVM missed 1 of 42 malignant cases, while Random Forest missed 3 and XGBoost missed 4.
- The test set is small (114 samples), so a difference of a few cases is not strong evidence that one model is truly better; results may change with a different random split.

## Limitations
- Small dataset (569 samples) and a single train/test split (cross-validation would give a more reliable comparison).
- No hyperparameter tuning was performed.
- This project is for educational purposes and is not a medical diagnostic tool.

## Technologies
Python, pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Seaborn, Google Colab

## How to Run
1. Open `Disease_Prediction.ipynb` in Google Colab (or Jupyter).
2. Run all cells (`Runtime → Run all`). The dataset is bundled with `scikit-learn`, so no manual download is needed.

Or locally:

    pip install numpy pandas matplotlib seaborn scikit-learn xgboost
    jupyter notebook Disease_Prediction.ipynb

## Repository Structure

    CodeAlpha_DiseasePrediction/
    ├── Disease_Prediction.ipynb
    └── README.md

## Possible Improvements
- Use k-fold cross-validation for a more robust comparison.
- Tune hyperparameters with `GridSearchCV`.
- Try other datasets from the same task (Heart Disease, Diabetes).
- Analyze feature importance to see which measurements drive the predictions.
