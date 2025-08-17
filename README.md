# Traditional_Machine_Learning_Projects
This repository contains the major traiditional machine learning projects that I had completed as a part of **"Advanced Machine Learning"** course.

# Census Income Prediction – Decision Trees & Random Forests


**Core Skills:** 'Supervised Learning', Decision Trees, Random Forests, Feature Engineering, Model Evaluation  

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

🚀 **Relevance to Job Roles:**  
This project mimics real-world classification problems where categorical and numerical features must be engineered, and trade-offs between model complexity, interpretability, and accuracy are evaluated. Skills like **feature preprocessing, model tuning, and performance validation** are directly applicable to roles in **Data Science, ML Engineering, and Applied AI Research**.
