# 📘 Machine Learning Classification Project  
*A multi-dataset supervised learning analysis using consistent preprocessing pipelines, custom feature selection, and multiple classical ML models.*

---

## 🚀 Project Overview  
This project evaluates multiple machine learning models across **four different classification datasets** (`TrainData1–4.txt`).  
Each notebook follows a consistent workflow:

- Load high-dimensional tabular data  
- Replace placeholder missing values (`1e+99 → NaN`)  
- Apply a custom preprocessing pipeline (median imputation + scaling + variance filtering)  
- Train and evaluate several classical ML models  
- Use train/test splits and stratified validation  
- Generate classification reports  
- Produce predictions on the provided test sets  

The datasets vary significantly in sample size and feature count, requiring different modeling choices and tuning strategies.

---

## 📁 Project Contents  
```
notebooks/
│── ml_project1.ipynb
│── ml_project2.ipynb
│── ml_project3.ipynb
│── ml_project4.ipynb

data/
│── TrainData1.txt
│── TrainData2.txt
│── TrainData3.txt
│── TrainData4.txt
│── TestData1.txt
│── TestData2.txt
│── TestData3.txt
│── TestData4.txt
│── Label1.txt
│── Label2.txt
│── Label3.txt
│── Label4.txt


# 🧹 Preprocessing Pipeline

All notebooks use a consistent manual preprocessing sequence:

### **1️⃣ Replace invalid numeric placeholders**
```python
df.replace(1.000000e+99, np.nan, inplace=True)
df_test.replace(1.000000e+99, np.nan, inplace=True)
```

### **2️⃣ Median imputation + standard scaling**
Pipeline combines:

- `SimpleImputer(strategy="median")`
- `StandardScaler()`

### **3️⃣ Custom Feature Selector**

All notebooks include this class:

```python
class Variance_Covariance_Threshold(BaseEstimator, TransformerMixin):
    def __init__(self, var_threshold=0.0, cov_threshold=1.0):
        self.var_threshold = var_threshold
        self.cov_threshold = cov_threshold

    def fit(self, X, y=None):
        X_values = np.asarray(X)
        var = X_values.var(axis=0)
        to_keep = var > self.var_threshold
        self.to_keep_idx_ = np.where(to_keep)[0]
        return self

    def transform(self, X):
        X_values = np.asarray(X)
        return X_values[:, self.to_keep_idx_]
```

This is applied before training.

---

# 🤖 Models Used 
Across all projects consistently trained:

- **Logistic Regression**
- **SVM (SVC)**
- **Random Forest**
- **MLPClassifier**
- **KNN**
- **Gaussian Naive Bayes**

Evaluation used:

```python
from sklearn.metrics import accuracy_score, classification_report
```

---

# 📊 Dataset Characteristics

### **Project 1**
- Shape: `df.shape` / `df_test.shape`
- Used Logistic Regression as the final model
- Ran repeated stratified K-fold cross-validation

### **Project 2**
- Very high dimensional  
- Logistic Regression and RandomForest performed best  

### **Project 3**
- 2547 samples × 112 features (9 classes)  
- Multiple models tested  
- MLPClassifier consistently highest accuracy  

### **Project 4**
- 1119 samples × 11 features  
- RandomForest chosen for hyperparameter tuning  
- Best params:

```
n_estimators=200
max_depth=20
max_features="sqrt"
min_samples_split=2
min_samples_leaf=1
```

---

# ▶️ How to Run the Project

## **1️⃣ Clone the repository**
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

## **2️⃣ Create & activate a virtual environment**

### Windows:
```bash
python -m venv venv
venv\Scriptsctivate
```

### macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

## **3️⃣ Install dependencies**
```bash
pip install -r requirements.txt
```

## **4️⃣ Launch Jupyter Notebook**
```bash
jupyter notebook
```

Open any notebook:

- `ml_project1.ipynb`
- `ml_project2.ipynb`
- `ml_project3.ipynb`
- `ml_project4.ipynb`

Run all cells top-to-bottom.

---

# 🔮 Making Predictions 

Every notebook ends with something like:

```python
preds = pipe_log.predict(df_test)
probs = pipe_log.predict_proba(df_test)
print("Predictions:", preds)
```

Or for RandomForest:

```python
preds = pipe_rf.predict(df_test)
```

---

---

# 📝 Summary of Findings

| Dataset | Feature Count | Samples | Best Model | Notes |
|--------|---------------|---------|------------|-------|
| **1** | ~3312 | 150 | Logistic Regression | CV accuracy strong despite overfitting concerns |
| **2** | ~9182 | 100 | Logistic Regression / RF | Feature selection improved F1 |
| **3** | 112 | 2547 | MLPClassifier | Only MLP exceeded 80% accuracy |
| **4** | 11 | 1119 | Random Forest | Best tuned via GridSearchCV |

---
