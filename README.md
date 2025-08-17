# Traditional_Machine_Learning_Projects
This repository contains the major traiditional machine learning projects that I had completed as a part of **"Advanced Machine Learning"** course.



## 1️⃣ [Census Income Prediction – Decision Trees & Random Forests](https://github.com/muhammadfarhan720/Traditional_Machine_Learning_Projects/tree/main/Random_Forest_Decisiong_Tree)


**Core Skills:** `Supervised Learning`, `Decision Trees`, `Random Forests`, `Feature Engineering`, `Model Evaluation`  

## Project Overview
The goal of this project was to predict whether an individual received dividend income (`HAS_DIV`) using census data.  
I implemented **decision trees** and **random forests** to evaluate predictive performance using both **binarized** and **continuous features**.  

This project provided hands-on experience with:
- Entropy-based **information gain** for feature selection  
- Building **decision tree classifiers** with scikit-learn  
- **Comparing criteria** (`entropy` vs `gini`) and tuning tree depth  
- Evaluating **confusion matrices** and **cross-validation scores**  
- Scaling continuous variables with `MinMaxScaler`  
- Experimenting with **RandomForestClassifier hyperparameters**:  
  - `n_estimators`, `criterion`, `max_depth`, `min_samples_leaf`, `bootstrap`

## Tools & Libraries
- **Python** (scikit-learn, pandas, numpy)  
- **Visualization**: `pydotplus`, `graphviz` for tree plotting  
- **Model Evaluation**: `sklearn.metrics.confusion_matrix`, `sklearn.model_selection.cross_val_score`  

## Key Learnings
- **Feature Engineering:** Created binary predictors (`AGI_BIN`, `A_AGE_BIN`, `WKSWORK_BIN`) and compared them with continuous counterparts.  
- **Model Training:** Implemented `DecisionTreeClassifier` with different depths (`max_depth=3,4,5,10`).  
- **Performance Analysis:** Observed that continuous features gave better accuracy and more balanced confusion matrices than binarized features.  
- **Random Forests:** Tuned `RandomForestClassifier` (e.g., `n_estimators=1000`, `max_depth=10–20`, `min_samples_leaf=1–100`) to achieve the best test accuracy (~67%).  
- **Business Insight:** Demonstrated how ensemble methods stabilize results compared to single decision trees.  

## Commands & APIs Practiced
- `pd.read_excel()` for structured data handling  
- `train_test_split()` for dataset splitting  
- `DecisionTreeClassifier()` & `RandomForestClassifier()` for supervised learning  
- `confusion_matrix()` & `cross_val_score()` for performance metrics  
- `export_graphviz()` & `pydotplus` for tree visualization  

---

## 2️⃣ [Multilayer Perceptron Regression & Classification on Census Data](https://github.com/muhammadfarhan720/Traditional_Machine_Learning_Projects/tree/main/MLP%20Regressor)

**Core Skills:** `Supervised Learning`, `Neural Networks`, `MLP Regression & Classification`, `Regularization Techniques`, `LinearDiscriminantAnalysis1, `Hyperparameter Tuning`, `Model Evaluation`, `TensorFlow/Keras Implementation`

### Project Overview
- Applied **Multilayer Perceptrons (MLPs)** to census datasets for both regression and classification tasks.  
- **Regression:** Predicted continuous `HDIVVAL` from features like AGI, age, sex, and work weeks using scikit-learn’s `MLPRegressor`.  
- Experimented with **hidden nodes (3–6)** and **L2 regularization (0–0.01)** to study effects on performance.  
- **Classification:** Predicted binary `F_BIN` from raw features, optimizing `MLPClassifier` to achieve ~60% accuracy.  
- Validated maximum achievable accuracy using **independent Linear Discriminant Analysis (LDA)**.  
- Reimplemented regression in **TensorFlow/Keras**, comparing frameworks and selecting the optimal model (3 neurons) based on **loss curves and MSE**.  
- Demonstrated how **network complexity and regularization** influence generalization, weights, and real-world applicability for census income prediction.  

### Tools & Libraries
- **Python** (scikit-learn, pandas, numpy, matplotlib)  
- **Neural Networks:** `MLPRegressor`, `MLPClassifier` (scikit-learn); TensorFlow/Keras for custom models  
- **Visualization:** matplotlib (loss curves, weight vs. epoch plots, confusion matrices)  
- **Model Evaluation:** `mean_squared_error`, `accuracy_score`, `confusion_matrix`  
- **Preprocessing:** `MinMaxScaler`, `train_test_split`  

### Key Learnings
- **Regularization Effects:** Increasing L2 (0 → 0.01) optimized weights to smaller values, reduced test MSE, and improved generalization—especially in larger models (6 hidden nodes)—helping prevent overfitting.  
- **Network Size Impact:** Larger hidden layers (4–6 nodes) increased complexity and initial MSE but benefited from regularization; smaller networks (3 nodes) trained faster but risked underfitting.  
- **Classification Optimization:** Tuned `MLPClassifier` (solver='sgd', alpha=0.001, hidden_size=3) to reach ~60% accuracy; validated via **LDA**, confirming dataset’s max achievable accuracy (~60%).  
- **TensorFlow vs. scikit-learn:** Implemented regression in TensorFlow, noting similar MSE across 3–6 neuron networks. Chose **3-neuron model** for simplicity and low validation loss, and highlighted framework differences in training history tracking.  
- **Performance Analysis:** Loss curves often showed validation loss < training loss due to small dataset size; **regularization stabilized training**, reflecting real-world ML trade-offs.  
- **Business Insight:** Demonstrated how MLPs can predict income-related features from census data, with **regularization ensuring robust, generalizable models** for policy or economic analysis.  

### Commands & APIs Practiced
- `pd.read_excel()` – load census datasets  
- `train_test_split()` – dataset splitting (e.g., test_size=0.3–0.4)  
- `MinMaxScaler()` – feature/target normalization  
- `MLPRegressor()` & `MLPClassifier()` – building/training MLPs (`partial_fit()` for epoch-wise training history)  
- `mean_squared_error()`, `accuracy_score()`, `confusion_matrix()` – evaluation metrics  
- `model.fit()`, `history.history` – TensorFlow/Keras training & loss visualization  
- `LinearDiscriminantAnalysis()` – independent accuracy estimation (baseline check with LDA)  

---
