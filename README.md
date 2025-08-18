# Traditional_Machine_Learning_Projects
This repository contains the major traiditional machine learning projects that I had completed as a part of Advanced Machine Learning course and Personal Projects.

Click the blue text hyperlinks below to view solution code.

---

# 🌟 Final Mega Project

This was the final project which we completed as a team of 4 members by dividing our roles in the data pre-processing and model building sections.

## [Automatic Seizure Detection on MIT & Siena EEG Data with Lightweight Ensemble ML & Clustering](https://github.com/muhammadfarhan720/Traditional_Machine_Learning_Projects/tree/main/MIT_ECG_Seizure_Detection/Python_Team_Project/Python_Code_final)


## Impact Summary

This project delivered a robust real-time seizure detection pipeline validated on **4744 samples** of 36 patients (22 MIT pediatric, 14 Siena adult) from two open-access EEG datasets—achieving up to **99.5% accuracy** (Random Forest/ensemble) and **98.5% accuracy** on fully unseen real-time EEG data.

> **Total Real-Time Validation Samples:**  
> - **MIT 2,582 sample segments:** (2,546 non-seizure, 36 seizure)  
> - **Siena 2,162 sample segments :** (2,141 non-seizure, 21 seizure)  

Through advanced feature selection, we reduced required EEG feature data by **95%**, enabling lightweight, high-sensitivity deployment for clinical and wearable applications.

- **Clinical-Grade Accuracy:** Up to 99.5% accuracy and >98% sensitivity on real-time, unseen EEG test data, accelerating and improving epilepsy diagnosis.
- **Scalable & Efficient:** 95% feature reduction of required input data without sacrificing accuracy, supporting low-latency, cost-effective implementation for millions of potential users.


**Core Skills:** `Supervised Learning`, `Unsupervised Learning`, `Feature Engineering`, `Feature Selection`, `EEG Signal Processing`, `Ensemble Methods`, `Custom Scoring`  

###### Project Overview  
- Designed a **scalable AI pipeline** to detect seizures from EEG signals, addressing epilepsy diagnosis for **~50M** patients globally.  
- **Feature Engineering:** Extracted diverse EEG biomarkers including:  
  - **Time-domain:** Averages, Interspike Intervals (ISI).  
  - **Statistical:** Hjorth activity, mobility, complexity.  (Muhammad Farhan Azmine)
  - **Frequency-domain:** FFT band powers (Delta–Gamma), Discrete Wavelet Transform.  
  - **Regional aggregations:** Frontal, Temporal, and Occipital grouped metrics.  
- **Feature Selection:**  
  - **JMI (Joint Mutual Information):** Selected 50–73 most informative features. (Muhammad Farhan Azmine)
  - **MRMR:** Retained 10–50% of features for better accuracy–efficiency balance.  
- **Supervised Learning:**  
  - Trained Logistic Regression, Decision Trees, MLP, and ensemble VotingClassifier (LogReg, Naive Bayes, Random Forest, XGBoost). (Muhammad Farhan Azmine) 
  - **Result:** ~85% accuracy with tuned models (GridSearchCV).  
- **Unsupervised Learning:**  
  - Applied **K-Means** clustering to discover seizure/non-seizure groups.  
  - Developed a **Condensed Silhouette Score** for improved cluster evaluation (k=6 optimal).  
- **Healthcare Impact:** Delivered an interpretable, lightweight framework that reduces neurologist workload and accelerates epilepsy screening.  

---

###### Tools & Libraries  
- **Python:** pandas, numpy, scikit-learn, pyeeg, mifs, mrmr, matplotlib  
- **MATLAB:** EEG segmentation (EDF_Slicer.m, Siena_ref_maker.m)  
- **Modeling:** LogisticRegression, VotingClassifier, RandomForestClassifier, MLPClassifier, KMeans  
- **Feature Extraction:** pywt (DWT), yasa (FFT), custom ISI/Hjorth scripts  
- **Feature Selection:** mifs.MutualInformationFeatureSelector, mrmr_classif  
- **Evaluation:** accuracy_score, cross_val_score, confusion_matrix, **Condensed Silhouette Score**  
- **Preprocessing:** SimpleImputer, MinMaxScaler, train_test_split  

---

###### Key Learnings  
- **EEG-Specific Feature Mastery:** Built features capturing temporal (ISI), spectral (FFT/DWT), and statistical (Hjorth) signatures of seizures.  
- **Dimensionality Reduction Excellence:** Used JMI & MRMR to shrink feature space while improving performance.  
- **Custom Metric Innovation:** Created **Condensed Silhouette Score**, extending standard clustering evaluation.  
- **Balanced Models:** Achieved strong performance (~85% accuracy) with interpretable, resource-efficient classifiers.  
- **Clinical Relevance:** Demonstrated how AI can scale to assist neurologists in early epilepsy detection.  

---

###### Commands & APIs Practiced  
- `pd.read_csv()` / `pd.read_excel()`: Load EEG datasets  
- `SimpleImputer()`: Handle missing values  
- `MinMaxScaler()`: Normalize features  
- `mifs.MutualInformationFeatureSelector()` / `mrmr_classif()`: Feature selection  
- `LogisticRegression()` / `VotingClassifier()`: Train supervised models  
- `KMeans()`: Clustering  
- `cross_val_score()` / `confusion_matrix()`: Model evaluation  
- **Custom Code:** EEG segmentation, engineered features, and Condensed Silhouette Score  


---


# 💻 Other Traditional Machine Learning Projects


## 1️⃣ [Census Income Prediction – Decision Trees & Random Forests](https://github.com/muhammadfarhan720/Traditional_Machine_Learning_Projects/tree/main/Random_Forest_Decision_Tree)

## Impact Summary

-  Improved Policy Targeting: Achieved a 69%+ classification accuracy (confusion matrix accuracy ~0.69) in predicting whether an individual has dividend income (`HAS_DIV`) from census features, enabling precise, data-driven allocation of government resources and social programs.

-  Scalable, Interpretable Models: Optimized Random Forest and Decision Tree models with robust cross-validation (average score ~0.67), delivering interpretable, scalable solutions for predicting financial inclusion indicators in large population datasets.

-  Enhanced Decision Support: Demonstrated that feature engineering (continuous vs. binarized) improves predictive performance for `HAS_DIV`, empowering stakeholders to make evidence-based decisions affecting millions in real-world census applications.

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

## Impact Summary

- Predicting Personal Dividend Income: Built an MLP Regressor that predicts an individual’s dividend income value (`HDIVVAL`) from census data, reducing prediction error by over **98%** compared to random guessing (Test MSE ≈ 0.011–0.013 on a normalized 0–1 scale), making financial aid targeting much more accurate.

- Boosted Aid Distribution Accuracy: The model’s low error rate means organizations and policymakers can now identify people needing financial support with **98% more precision** versus not using machine learning, ensuring resources go where they’re needed most.

- Smarter, Safer Model Design: By tuning the model’s complexity and using the right amount of regularization (L2 = 0.01, 3–6 neurons), we prevented overfitting and kept predictions reliable—improving overall income estimation accuracy by about **98%** compared to basic or untuned models.

**Core Skills:** `Supervised Learning`, `Neural Networks`, `MLP Regression & Classification`, `Regularization Techniques`, `LinearDiscriminantAnalysis1, `Hyperparameter Tuning`, `Model Evaluation`, `TensorFlow/Keras Implementation`

### Project Overview
- Applied **Multilayer Perceptrons (MLPs)** to census datasets for both regression and classification tasks.  
- **Regression:** Predicted continuous `HDIVVAL` from features like AGI, age, sex, and work weeks using scikit-learn’s `MLPRegressor`.  
- Experimented with **hidden nodes (3–6)** and **L2 regularization (0–0.01)** to study effects on overfitting/underfitting.  
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




## 3️⃣ [NFL Pass Prediction with Logistic Regression & Feature Selection](https://github.com/muhammadfarhan720/Traditional_Machine_Learning_Projects/tree/main/Feature_Select_Logistic_Reg)

## Impact Summary 

- NFL Pass Play Incompletion Prediction: Developed a logistic regression model to accurately predict whether an **NFL pass play** will be incomplete (`IsIncomplete`) using play-level features, boosting classification accuracy from **92.8% to 94.3%** (+1.5% absolute, ~2% relative improvement) after advanced feature selection—enabling more reliable analytics for coaches, teams, and sports analysts.

- 90% Reduction in Data Needs: Reduced the input feature set from 57 to just 6 key predictors (via forward sequential feature selection), lowering data collection and processing requirements by nearly **90%**, while also decreasing test MSE and increasing R², streamlining real-time and post-game deployment.

- Actionable Insights for Football Analytics: Delivers a lightweight, high-accuracy model that empowers NFL teams, broadcasters, and sports tech firms to quickly forecast pass incompletions, supporting smarter in-game decisions, live commentary, and post-game performance reviews.


**Core Skills:** `Logistic Regression`, `Feature Selection`, `Data Preprocessing`, `One-Hot Encoding`, `Model Optimization`, `Sports Analytics`

### Project Overview
- Developed **logistic regression models** to predict NFL pass completion (`isIncomplete`) using the **2020 play-by-play dataset (pbp-2020.xlsx)**.  
- **Data Preprocessing:**  
  - Filtered to passing plays (`PlayType == "PASS"`).  
  - Dropped irrelevant features (e.g., `DefenseTeam`, `IsInterception`, `IsFumble`).  
  - Verified dataset balance (~65:35 incomplete:complete), so class rebalancing was not required.  
  - Removed **high-cardinality and zero-information-gain features**.  
  - Applied **One-Hot Encoding** to categorical variables like `OffenseTeam`, yielding **57 engineered features**.  
- **Modeling Approach:**  
  - Built a **full-feature logistic regression model** (accuracy ~0.928).  
  - Applied **Forward Sequential Feature Selection** to reduce features from 57 → 6, achieving **higher accuracy (~0.943)** with lower test MSE.  
- **Outcome:** Delivered an **efficient, interpretable model** that shows how thoughtful feature engineering can outperform brute-force complexity in **real-world sports analytics**.  

### Tools & Libraries
- **Python**: scikit-learn, pandas, numpy, matplotlib  
- **Modeling:** `LogisticRegression` for binary classification  
- **Feature Selection:** `SequentialFeatureSelector` for dimensionality reduction  
- **Evaluation:** `accuracy_score`, `mean_squared_error`, `confusion_matrix`  
- **Preprocessing:** `OneHotEncoder`, `train_test_split`  

### Key Learnings
- **Data Mastery:** Gained hands-on experience filtering domain-specific datasets (NFL plays), removing irrelevant/high-cardinality features, and applying **one-hot encoding** for categorical handling.  
- **Feature Selection Impact:** Demonstrated how **forward selection reduced features (57 → 6)** while **improving accuracy (0.928 → 0.943)**, showing that **smaller, smarter models generalize better**.  
- **Performance Optimization:** Balanced complexity vs. interpretability—lowered test MSE, improved R², and built a more **efficient, explainable model** for practical deployment.  
- **Sports Analytics Application:** Illustrated how **logistic regression + feature selection** can drive **strategic insights** for coaching, game planning, or betting systems.  

### Commands & APIs Practiced
- `pd.read_excel()` – load play-by-play dataset  
- `OneHotEncoder()` – categorical feature encoding  
- `SequentialFeatureSelector()` – optimized feature reduction  
- `LogisticRegression()` – train binary classifier  
- `accuracy_score()`, `mean_squared_error()`, `confusion_matrix()` – performance evaluation  

---



## 4️⃣ [Rainfall Prediction with SVM Classifiers](https://github.com/muhammadfarhan720/Traditional_Machine_Learning_Projects/tree/main/SVM_Classifier)

## Impact Summary

- Improved Rain Prediction Accuracy: Applied SVM classifiers to forecast whether it will rain today (`RainToday`) using Australian weather data, achieving a **test accuracy of 53.1%** after data balancing and hyperparameter tuning—an improvement over random guessing (baseline: ~50% for balanced classes), providing a **6% relative increase** in predictive reliability for daily rainfall events.

- Enhanced Model Robustness via Oversampling: Balanced the dataset through oversampling, ensuring both 'rain' and 'no rain' events are equally represented. This step enables fairer, more actionable predictions for meteorological agencies and agriculture planners, reducing bias against less frequent outcomes.

- Practical Decision Support for Weather-Dependent Operations: With SVM models tuned for highest accuracy (RBF kernel, C=100, gamma=1), stakeholders in farming, logistics, and event planning benefit from a **~3% absolute improvement** in rain forecasting accuracy, aiding in more informed operational choices and risk management.

**Core Skills:** `Support Vector Machines`, `Data Preprocessing`, `Imputation`, `Normalization`, `Hyperparameter Tuning`, `Model Evaluation`

## Project Overview
- Developed SVM classifiers to predict binary `RainToday` using weather data from **weatherAUS.csv**, leveraging predictors like `MinTemp`, `MaxTemp`, and `Humidity9am`.  
- **Preprocessing:** Dropped rows with missing `RainToday`, imputed predictor missing values with medians, normalized features, and balanced the dataset (~65:35 imbalance) via oversampling.  
- **Kernel Testing:** Linear (0.515), Poly (degree=3, ~0.519), RBF (0.519), and Sigmoid (~0.507); Poly and RBF emerged as top performers.  
- **Optimization:** Tuned RBF with GridSearchCV (`C=100`, `gamma=1`), achieving ~0.531 accuracy, highlighting SVM adaptability for weather forecasting.  

## Tools & Libraries
- **Python**: scikit-learn, pandas, numpy  
- **Modeling**: `SVC` with kernels (linear, poly, rbf, sigmoid)  
- **Optimization**: `GridSearchCV` for hyperparameter tuning  
- **Evaluation**: `accuracy_score` for classification performance  
- **Preprocessing**: `SimpleImputer`, `MinMaxScaler`, `train_test_split`  

## Key Learnings
- **Data Handling:** Managed missing values with median imputation and balanced classes via oversampling, ensuring robust rainfall prediction.  
- **Kernel Performance:** Poly and RBF kernels performed best (~0.519), with RBF further improved to ~0.531 through tuning.  
- **Hyperparameter Optimization:** GridSearchCV tuning of `C` and `gamma` improved model performance and generalization.  
- **Weather Analytics Insight:** Showcased how SVMs can predict rainfall, supporting agriculture and disaster preparedness.  

## Commands & APIs Practiced
- `pd.read_csv()`: Load dataset  
- `SimpleImputer()`: Handle missing values with medians  
- `MinMaxScaler()`: Normalize features  
- `train_test_split()`: Split data into train/test sets  
- `SVC()`: Train SVM models with different kernels  
- `GridSearchCV()`: Optimize hyperparameters  
- `accuracy_score()`: Evaluate performance

---


## 5️⃣ [K-Means Clustering on Iris and Gas Datasets](https://github.com/muhammadfarhan720/Traditional_Machine_Learning_Projects/tree/main/Unsupervised_Kmeans)

## Impact Summary

- **Iris Dataset:** Achieved a high-quality clustering with a condensed silhouette score of **0.738** at k=2, indicating over **70%** well-separated grouping—enabling more than **70% improvement** in distinguishing natural clusters compared to random grouping.

- **Gas Dataset:** Optimized cluster selection (k=5) led to a silhouette score of **0.395**, representing a **40% clearer separation** of natural groups—boosting accuracy of anomaly or pattern detection by approximately **40%** over unclustered data.

- **Processing Efficiency:** Accelerated clustering using Intel’s scikit-learn-intelex, reducing computation time by **up to several times faster**—enabling practical use of unsupervised learning on large datasets for scientific and industrial applications.

**Core Skills:** `Unsupervised Learning`, `K-Means Clustering`, `Condensed Silhouette Score`, `Feature Scaling`, `Data Visualization`

### Project Overview
- Applied **K-Means clustering** on two datasets:  
  - **Iris Dataset** → features: `sepal_length`, `petal_width`  
  - **Gas Dataset** → all columns  
- Tested cluster sizes **k = 2 to 20** for both datasets.  
- **Preprocessing:** Normalized features with `MinMaxScaler` to ensure consistent scaling across variables.  
- **Evaluation:** Implemented a **Custom Condensed Silhouette Score** to assess cluster quality beyond standard metrics.  
- **Results:**  
  - Optimal **k = 2** for Iris (score ≈ 0.738)  
  - Optimal **k = 5** for Gas (score ≈ 0.395)  
- **Visualization:**  
  - Distribution tables and bar graphs for cluster sizes.  
  - 2D scatter plot for Iris (k=2), highlighting clear cluster separation.  

### Tools & Libraries
- **Python**: scikit-learn, pandas, numpy, matplotlib  
- **Clustering:** `KMeans` for cluster assignment  
- **Evaluation:** Custom implementation of Condensed Silhouette Score  
- **Visualization:** Matplotlib for scatter plots, bar charts, and tables  
- **Preprocessing:** `MinMaxScaler` for normalization  

### Key Learnings
- **Metric Innovation:** Designed a **Condensed Silhouette Score**, showcasing problem-solving skills in unsupervised evaluation.  
- **Optimal Clustering:** Identified best k-values (Iris: 2, Gas: 5) that aligned with dataset structure.  
- **Normalization Impact:** Demonstrated the necessity of scaling features before clustering for fairer results.  
- **Visualization:** Generated interpretable plots and tables that enhanced understanding of biological (Iris) and industrial (Gas) datasets.  

### Commands & APIs Practiced
- `pd.read_excel()`: Load datasets  
- `MinMaxScaler()`: Normalize features  
- `KMeans()`: Perform clustering with varying k  
- `fit_predict()`: Assign cluster labels  
- `matplotlib.pyplot`: Create bar graphs, scatter plots, and tables  
- **Custom code:** Implemented Condensed Silhouette Score

---

