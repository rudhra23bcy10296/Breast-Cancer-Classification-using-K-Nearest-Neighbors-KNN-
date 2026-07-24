# AI-ML Assignment 4: Breast Cancer Classification using K-Nearest Neighbors (KNN)

## Objective
The primary objective of this project is to develop and evaluate a **K-Nearest Neighbors (KNN)** machine learning classification model to accurately predict whether a breast tumor is **Malignant (M)** or **Benign (B)** based on diagnostic feature measurements.

---

## Dataset Link
- **Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Dataset
- **Kaggle Link:** [https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

*Note: Per assignment submission instructions, the raw dataset file (`data.csv`) is excluded from the repository via `.gitignore`.*

---

## Repository Structure
```
Assignment 4/
│── Assignment-4.ipynb    # Main Jupyter Notebook with code, outputs & analysis
│── README.md             # Project documentation & assignment details
└── .gitignore            # Excludes data.csv and temporary cache files
```

---

## Libraries Used
- **Pandas**: For data loading, manipulation, inspection, and missing value checks.
- **NumPy**: For numerical array operations.
- **Scikit-learn (`sklearn`)**:
  - `train_test_split`: For splitting the dataset into 80% training and 20% testing sets.
  - `StandardScaler`: For standardizing feature values (zero mean, unit variance).
  - `LabelEncoder`: For encoding categorical target labels (`B` $\rightarrow$ `0`, `M` $\rightarrow$ `1`).
  - `KNeighborsClassifier`: For building and training the KNN classification model ($K=5$).
  - `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`, `classification_report`: For model evaluation metrics.

---

## Methodology

### Task 1: Data Understanding
- Loaded the dataset using `pd.read_csv()`.
- Inspected the first 5 records (`df.head()`).
- Identified **30 numerical features** (e.g., `radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`, `smoothness_mean`, `compactness_mean`, `concavity_mean`, `concave points_mean`, etc.).
- Identified the **target variable** (`diagnosis`: 357 Benign, 212 Malignant).
- Generated full dataset info (`df.info()`) and summary statistics (`df.describe()`).

### Task 2: Data Preprocessing
- **Missing Value Check:** Identified missing values (column `Unnamed: 32` contained 569 missing values).
- **Feature Selection / Cleanup:** Dropped unnecessary non-predictive columns (`id` and empty artifact `Unnamed: 32`).
- **Target Encoding:** Encoded string labels into binary numbers (`B` $\rightarrow$ `0` [Benign], `M` $\rightarrow$ `1` [Malignant]).
- **Dataset Splitting:** Partitioned the data into **80% Training Set (455 samples)** and **20% Testing Set (114 samples)** using stratified splitting (`random_state=42`).
- **Feature Scaling:** Applied `StandardScaler` to ensure all 30 features have zero mean and unit variance, preventing high-magnitude features (e.g., `area_mean` $\sim 1000$) from disproportionately dominating distance metrics in KNN.

### Task 3: Model Development
- Trained a **K-Nearest Neighbors (KNN) Classifier** with initial hyperparameter $K = 5$.
- Generated class label predictions (`y_pred`) on the 114 test samples.

---

## Results

### Model Evaluation Metrics ($K = 5$)

| Metric | Score | Percentage |
| :--- | :---: | :---: |
| **Accuracy** | `0.9561` | **95.61%** |
| **Precision** | `0.9744` | **97.44%** |
| **Recall** | `0.9048` | **90.48%** |
| **F1-Score** | `0.9383` | **93.83%** |

### Confusion Matrix

| | Predicted Benign (0) | Predicted Malignant (1) |
| :--- | :---: | :---: |
| **Actual Benign (0)** | **71** *(True Negative)* | **1** *(False Positive)* |
| **Actual Malignant (1)** | **4** *(False Negative)* | **38** *(True Positive)* |

### Model Performance Observations
1. **High Overall Accuracy:** The KNN classifier achieved an overall accuracy of **95.61%** (109 out of 114 test cases correctly classified), proving highly effective at distinguishing malignant from benign tumors.
2. **High Precision for Malignant Predictions:** With a precision of **97.44%**, when the model predicts a tumor is malignant, it is correct nearly 97.5% of the time, yielding only 1 False Positive.
3. **Clinical Implication of Recall:** The recall for malignant tumors is **90.48%** (4 False Negatives). In cancer diagnostics, minimizing False Negatives (missing cancer) is critical to ensure patients receive timely treatment. Hyperparameter tuning or probability thresholding can be used to further improve recall in clinical settings.

---

## Conclusion

The K-Nearest Neighbors (KNN) classification model demonstrates strong efficacy in predicting breast cancer tumor malignancy, achieving an overall accuracy of 95.61% and an F1-score of 93.83% on the test set. Feature scaling via StandardScaler was fundamental to this performance because KNN relies entirely on Euclidean distance metrics between data points; without standardization, high-magnitude features like tumor area would overwhelmingly dominate low-magnitude features like smoothness, severely degrading classification accuracy. However, a key limitation of the KNN algorithm is its high computational latency during inference, as it requires calculating distances to all training instances for every new prediction, making it memory-intensive and slower on larger diagnostic datasets. Overall, KNN serves as an effective baseline for diagnostic classification when features are properly normalized.

---

## How to Run

1. Ensure Python 3.x and required dependencies are installed:
   ```bash
   pip install pandas numpy scikit-learn jupyter
   ```
2. Open and execute `Assignment-4.ipynb` using Jupyter Notebook or VS Code:
   ```bash
   jupyter notebook Assignment-4.ipynb
   ```
