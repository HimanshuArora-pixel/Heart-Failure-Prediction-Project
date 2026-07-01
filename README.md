# Heart Failure Prediction - Tree Ensembles

## 📌 Project Overview
This repository contains a robust, portfolio-grade classification model built to predict heart disease. Inspired by the **DeepLearning.AI Machine Learning Specialization** (Course 2: Advanced Learning Algorithms), this project demonstrates a deliberate, step-by-step model engineering lifecycle. The primary focus is on understanding and controlling variance, bias, and hyperparameter tuning through Tree Ensembles.

## 📊 Dataset
* **Source/Name:** `Dataset.csv` (Clinical patient data)
* **Size:** 918 rows
* **Target Variable:** `HeartDisease` (Binary Classification)

## ⚙️ Phase 1: Data Preprocessing
To ensure the models interpret the data correctly without assuming false ordinal relationships, the following preprocessing steps were applied:
* **One-Hot Encoding:** Categorical variables (`Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope`) were converted into binary numerical features using `pd.get_dummies`.
* **Feature Selection:** Isolated the target variable and retained all encoded columns as predictors.
* **Data Splitting:** Applied `train_test_split` with an 80/20 ratio and `random_state=55` to guarantee reproducible results across training runs.

## 🧠 Phase 2: Model Engineering Lifecycle
The project progresses from a simple baseline to a highly tuned ensemble, illustrating the practical machine learning lifecycle.

### 1. Decision Tree (The Baseline)
* **Objective:** Establish a baseline performance and find the complexity "sweet spot."
* **Tuning:** Iteratively tested `max_depth` to prevent the model from memorizing training data.
* **Optimal Constraint:** `max_depth=4`.

### 2. Random Forest (Variance Reduction)
* **Objective:** Upgrade to an ensemble approach (Bagging) using 100 estimators to reduce variance.
* **Observation:** Initial tests showed severe overfitting (100% training accuracy).
* **Tuning:** Applied strict regularization constraints (`min_samples_split=10`, `max_depth=16`) to force generalized pattern recognition over noise memorization.

### 3. XGBoost (The Final Boss)
* **Objective:** Implement Gradient Boosting to sequentially correct errors from previous trees.
* **Original Tuning & Optimization:** * Engineered custom iterative loops to systematically test hyperparameters rather than relying on default values.
  * Evaluated `n_estimators` up to 1,500 to find the exact point of diminishing returns.
  * Fine-tuned `learning_rate` (0.01) and `max_depth` (4) to specifically combat overfitting and prioritize generalized performance.

## 🏆 Phase 3: Final Benchmark Results
The iterative tuning and regularization process yielded the following comparative metrics:

| Model | Training Accuracy | Test Accuracy |
| :--- | :---: | :---: |
| **Decision Tree (Baseline)** | 88.28% | 86.96% |
| **Random Forest (Regularized)** | 93.05% | 89.13% |
| **XGBoost (Tuned)** | **95.50%** | **89.67%** |

## 📈 Phase 4: Model Evaluation & Feature Insights
Beyond basic accuracy, the final XGBoost model was evaluated to understand its decision-making process and error types:
* **Confusion Matrix Analysis:** The model successfully identified 89 true positive cases (Heart Disease) and 75 true negative cases. It exhibited a low false-positive rate (6 cases), though it produced 14 false negatives—a critical metric to monitor and minimize in medical diagnostic contexts.
* **Feature Importance:** Extracting the internal weights of the model revealed that **Cholesterol** is overwhelmingly the most significant predictor (importance score: 1302.0). This was followed by Maximum Heart Rate (`MaxHR`), Age, and `Oldpeak`, indicating that continuous physiological metrics drove the model's predictive power much more heavily than categorical symptoms.

## 🛠️ Tech Stack
* **Language:** Python
* **Environment:** Jupyter Notebook / Anaconda
* **Libraries:** Pandas, Numpy, Scikit-Learn, XGBoost

## 👨‍💻 Author
**Himanshu Arora**
* B.Tech Information Technology (Class of 2029) @ IIIT Allahabad
* Member, Club of Programmers (AI/ML Wing)
