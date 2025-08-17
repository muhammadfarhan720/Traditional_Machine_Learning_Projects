import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
import numpy
import numpy as np
import csv
import warnings
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns
import pandas as pd
import sklearn.tree as tree
import sklearn.impute as impute
import sklearn.model_selection as modelsel
import sklearn.metrics as metrics
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import r2_score
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import graphviz
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import pydotplus
import collections

class functions:



    def calc_IG(dirPath, fileName):
        # https://www.youtube.com/watch?v=81JSbXZ26Ls
        # https://stackoverflow.com/questions/64343345/how-to-select-best-features-in-dataframe-using-the-information-gain-measure-in-s
        # fileName = fileName # Add file name with a '' like 'chnl_band_power_Siena.csv'

        df = pd.read_excel(dirPath + fileName)
        df.dropna(inplace=True)
        targetName = 'target'

        X = df.drop([targetName], axis=1)
        y = df[targetName]
        ranSeed = 24060
        # To avoid overfitting
        X_train, X_test, y_train, y_test = modelsel.train_test_split(X, y, test_size=0.3, random_state=ranSeed)

        mutual_info = mutual_info_classif(X_train, y_train)
        mutual_info = pd.Series(mutual_info)
        mutual_info.index = X_train.columns
        m = mutual_info.sort_values(ascending=False)
        print('Info Gain for all features:\n', m)

        """
        # select the top best features
        sel_top_cols = SelectKBest(mutual_info, k =10)
        sel_top_cols.fit(X_train, y_train)
        n = X_train.columns[sel_top_cols.get_support()]
        print('Selected Featurees are:\n', n)
        """

        # Select the top 5% of the features
        sel_top_cols_by_per = SelectPercentile(mutual_info_classif, percentile=2)
        sel_top_cols_by_per.fit(X_train, y_train)
        features = X_train.columns[sel_top_cols_by_per.get_support()]
        print('Top 5% of the features in number:\n', len(features))
        print('Top 5% of the features are:\n', features)


    def mlp_clf(dirPath, fileName, hl, epochs, alpha, randomSeed):

        """
        # Siena top 2% Feature
        df = pd.read_excel(dirPath + fileName, usecols=['T4_variance', 'F10_variance', 'O1_ISI', '2_ISI', 't4_dwtcavar',
                                                        't4_dwtcarms', 't5_dwtcdvar', 't6_dwtcdvar', 'f7_dwtcdrms',
                                                        'Occipital_ISImed', 'Occipital_ISImean', 'Temporal_ISImed',
                                                        'Temporal_ISImean', 'Temporal_dwtcArmsmean', 'Temporal_dwtcAvarmean', 'target'])






        # Siena top 5% Feature
        df = pd.read_excel(dirPath + fileName, usecols=['T4_variance', 'F10_variance', 'O1_ISI', 'T3_ISI', 'T5_ISI', 'P4_ISI',
                                                        'T6_ISI', '2_ISI', 't5_dwtcavar', 't4_dwtcavar', 'f10_dwtcavar',
                                                        'f8_dwtcarms', 't4_dwtcarms', 't6_dwtcarms', 'f10_dwtcarms',
                                                        't5_dwtcdvar', 't6_dwtcdvar', 'f7_dwtcdrms', 't6_dwtcdrms', 'T4_Theta',
                                                        'Temporal_Thetamed', 'Temporal_variancemed', 'Temporal_variancemean',
                                                        'Occipital_ISImed', 'Occipital_ISImean', 'Temporal_ISImed',
                                                        'Temporal_ISImean', 'CentPar_ISImed', 'CentPar_ISImean',
                                                        'Temporal_dwtcArmsmean', 'FrontCent_dwtcArmsmed',
                                                        'FrontCent_dwtcArmsmean', 'Occipital_dwtcDrmsmed',
                                                        'Occipital_dwtcDrmsmean', 'Temporal_dwtcAvarmed',
                                                        'Temporal_dwtcAvarmean', 'Temporal_dwtcDvarmed', 'target'])

        # Siena top 10% Feature
        df = pd.read_excel(dirPath + fileName, usecols=['T5_variance', 'F4_variance', 'C4_variance', 'F8_variance',
                                                        'T4_variance', 'FC6_variance', 'CP6_variance', 'F10_variance', 'O1_ISI',
                                                        'T3_ISI', 'T5_ISI', 'Cp5_ISI', 'F9_ISI', 'Pz_ISI', 'P4_ISI', 'O2_ISI',
                                                        'T6_ISI', '2_ISI', 'T5_ZC', 't5_dwtcavar', 'f8_dwtcavar', 't4_dwtcavar',
                                                        'fc6_dwtcavar', 'cp6_dwtcavar', 'f10_dwtcavar', 'cp5_dwtcarms',
                                                        'f8_dwtcarms', 't4_dwtcarms', 't6_dwtcarms', 'fc6_dwtcarms',
                                                        'f10_dwtcarms', 'o1_dwtcdvar', 't5_dwtcdvar', 'cp5_dwtcdvar',
                                                        'f9_dwtcdvar', 't6_dwtcdvar', 'f3_dwtcdrms', 'o1_dwtcdrms',
                                                        'f7_dwtcdrms', 't5_dwtcdrms', 'cp5_dwtcdrms', 'f9_dwtcdrms',
                                                        'o2_dwtcdrms', 't6_dwtcdrms', 'F4_Theta', 'C4_Theta', 'T4_Theta',
                                                        'Temporal_Thetamed', 'Frontal_variancemean', 'Temporal_variancemed',
                                                        'Temporal_variancemean', 'CentPar_ZCmean', 'Parietal_ISImean',
                                                        'Occipital_ISImed', 'Occipital_ISImean', 'Temporal_ISImed',
                                                        'Temporal_ISImean', 'CentPar_ISImed', 'CentPar_ISImean',
                                                        'Frontal_dwtcArmsmed', 'Temporal_dwtcArmsmed', 'Temporal_dwtcArmsmean',
                                                        'FrontCent_dwtcArmsmed', 'FrontCent_dwtcArmsmean',
                                                        'Parietal_dwtcDrmsmean', 'Occipital_dwtcDrmsmed',
                                                        'Occipital_dwtcDrmsmean', 'Temporal_dwtcDrmsmed',
                                                        'Temporal_dwtcAvarmed', 'Temporal_dwtcAvarmean',
                                                        'Occipital_dwtcDvarmed', 'Occipital_dwtcDvarmean',
                                                        'Temporal_dwtcDvarmed', 'target'])




        # MIT Features top 2% Feature
        df = pd.read_excel(dirPath + fileName, usecols=['C3-P3_ISI', 'P3-O1_ISI', 'T7-P7_Theta', 'T7-P7_Alpha', 'P7-O1_Theta',
                                                        'TempPar_Thetamed', 'TempPar_Thetamean', 'TempPar_Alphamed',
                                                        'TempPar_dwtcArmsmed', 'TempPar_dwtcArmsmean', 'target'])





        # MIT RT Features top 5% Feature
        df = pd.read_excel(dirPath + fileName, usecols=['F7-T7_Delta', 'F7-T7_Theta', 'T7-P7_Delta', 'T7-P7_Theta',
                                                        'P7-O1_Theta', 'f7-t7_dwtcavar', 't7-p7_dwtcavar', 'fp1-f7_dwtcarms',
                                                        'f7-t7_dwtcarms', 't7-p7_dwtcarms', 'fp1-f3_dwtcarms', 'fz-cz_dwtcarms',
                                                        'FrontTemp_Thetamed', 'FrontTemp_Thetamean', 'FrontTemp_dwtcArmsmed',
                                                        'FrontTemp_dwtcArmsmean', 'FrontCent_dwtcArmsmed',
                                                        'FrontTemp_dwtcAvarmed', 'FrontTemp_dwtcAvarmean',
                                                        'TempPar_dwtcAvarmed', 'TempPar_dwtcAvarmean', 'target'])

        """
        # Siena top 5% Grouped RT Data
        df = pd.read_excel(dirPath + fileName, usecols=['F9_Theta', 'F9_Gamma', 'T6_Gamma', 'Fc6_Theta', 'Fc6_Alpha', 'f7_ISI',
                                                        'cp5_ISI', 'f9_ISI', 'p4_ISI', 't6_ISI', 'cp2_ISI', 'cp6_ISI',
                                                        'f10_ISI', 'cp5_dwtcdvar', 't6_dwtcdvar', 'p4_dwtcdrms', 't6_dwtcdrms',
                                                        'f10_dwtcdrms', 'CentPar_Gammamed', 'CentPar_Gammamean',
                                                        'Parietal_ISImed', 'Parietal_ISImean', 'CentPar_ISImed',
                                                        'CentPar_ISImean', 'Parietal_dwtcDrmsmed', 'Parietal_dwtcDrmsmean',
                                                        'CentPar_dwtcDrmsmed', 'CentPar_dwtcDrmsmean', 'CentPar_dwtcDvarmed',
                                                        'CentPar_dwtcDvarmean', 'target'])
        """
        # MIT Features top 10% Feature
        df = pd.read_excel(dirPath + fileName, usecols=['F7-T7_variance', 'T7-P7_variance', 'P7-O1_variance', 'P3-O1_variance',
                                                        'f7-t7_dwtcavar', 't7-p7_dwtcavar', 'p7-o1_dwtcavar', 'f7-t7_dwtcarms',
                                                        't7-p7_dwtcarms', 'P7-O1_ISI', 'F3-C3_ISI', 'C3-P3_ISI', 'P3-O1_ISI',
                                                        'FZ-CZ_ISI', 'CZ-PZ_ISI', 'P4-O2_ISI', 'T8-P8_ISI', 'F7-T7_Theta',
                                                        'F7-T7_Alpha', 'T7-P7_Theta', 'T7-P7_Alpha', 'P7-O1_Theta',
                                                        'P7-O1_Alpha', 'P7-O1_Beta', 'C3-P3_Alpha', 'P3-O1_Theta',
                                                        'P3-O1_Alpha', 'P3-O1_Beta', 'FrontTemp_Thetamed',
                                                        'FrontTemp_Thetamean', 'TempPar_Thetamed', 'TempPar_Thetamean',
                                                        'ParOcc_Thetamean', 'TempPar_Alphamed', 'TempPar_Alphamean',
                                                        'ParOcc_Betamed', 'ParOcc_Betamean', 'FrontTemp_variancemed',
                                                        'ParOcc_variancemed', 'ParOcc_variancemean', 'ParOcc_ISImed',
                                                        'ParOcc_ISImean', 'CentPar_ISImed', 'CentPar_ISImean',
                                                        'TempPar_dwtcArmsmed', 'TempPar_dwtcArmsmean', 'ParOcc_dwtcArmsmean',
                                                        'TempPar_dwtcAvarmed', 'TempPar_dwtcAvarmean', 'ParOcc_dwtcAvarmed', 'target'])

        


        
        """
        clf = MLPClassifier(hidden_layer_sizes=hl, solver='adam',
                            activation='relu', tol=0.0000001, learning_rate='constant',
                            learning_rate_init=0.01, shuffle=False, max_iter=epochs,
                            random_state=randomSeed, alpha=alpha, validation_fraction=0.42, warm_start=True)

        targetName = 'target'
        X = df.drop([targetName], axis=1).to_numpy()

        y = df[targetName].to_numpy()

        # Examining and replacing missing values with imputer function using median option.
        imp = impute.SimpleImputer(strategy='median')
        imp.fit(X)
        X = imp.transform(X)

        # normalization
        Xscaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        # yscaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        X = Xscaler.fit_transform(X)
        # y = yscaler.fit_transform(y.reshape(-1, 1)).ravel()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=randomSeed)
        # Dropping target to train the model

        """
        # Parameter Tuning
        # https://panjeh.medium.com/scikit-learn-hyperparameter-optimization-for-mlpclassifier-4d670413042b
        mlp_gs = MLPClassifier(max_iter=100)
        parameter_space = {
            'hidden_layer_sizes': [(10, 30, 10), (20,)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant', 'adaptive'],
        }

        clf = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=5)
        clf.fit(X, y)  # X is train samples and y is the corresponding labels
        print('Best parameters found:\n', clf.best_params_)

        """

        clf.fit(X_train, y_train)

        # Measure model performance
        y_pred = clf.predict(X_test)
        print('############################## MLP Classifier ##################################')
        print(
            "SK: {} epochs, {} accuracy score on {} samples".format(clf.n_iter_, metrics.accuracy_score(y_test, y_pred),
                                                                    X_test.shape[0]))
        print('##################################################################################')


        # Source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html
        plot_confusion_matrix(clf, X_test, y_test)
        plt.show()
        """ 
        plt.plot(clf.loss_curve_)
        plt.title('training set loss')
        plt.ylabel('loss')
        plt.yscale("log")
        xlabel = "epochs (hl=" + str(hl) + ")"
        plt.show()
        """




    def decision_tree_classifier(dirPath, fileName):

        # MIT Features top 5% Feature
        df = pd.read_excel(dirPath + fileName,
                           usecols=['T7-P7_variance', 't7-p7_dwtcavar', 'p7-o1_dwtcavar', 'f7-t7_dwtcarms',
                                    't7-p7_dwtcarms', 'F3-C3_ISI', 'C3-P3_ISI', 'P3-O1_ISI',
                                    'F7-T7_Theta', 'F7-T7_Alpha', 'T7-P7_Theta', 'T7-P7_Alpha',
                                    'P7-O1_Theta', 'P7-O1_Alpha', 'TempPar_Thetamed', 'TempPar_Thetamean',
                                    'TempPar_Alphamed', 'TempPar_Alphamean', 'CentPar_ISImed',
                                    'CentPar_ISImean', 'TempPar_dwtcArmsmed', 'TempPar_dwtcArmsmean',
                                    'TempPar_dwtcAvarmed', 'TempPar_dwtcAvarmean', 'target'])
        """
        # Siena Features top 5% Feature
        df = pd.read_excel(dirPath + fileName,
                           usecols=['F4_variance', 'T4_variance', 'F10_variance', 'O1_ISI', 'T5_ISI',
                                    'T6_ISI', '2_ISI', 't5_dwtcavar', 't4_dwtcavar', 'f10_dwtcavar',
                                    'f8_dwtcarms', 't4_dwtcarms', 't6_dwtcarms', 'f10_dwtcarms',
                                    't5_dwtcdvar', 'cp5_dwtcdvar', 't6_dwtcdvar', 'f7_dwtcdrms',
                                    't6_dwtcdrms', 'T4_Theta', 'Temporal_Thetamed', 'Temporal_variancemed',
                                    'Temporal_variancemean', 'Occipital_ISImed', 'Occipital_ISImean',
                                    'Temporal_ISImed', 'Temporal_ISImean', 'CentPar_ISImed',
                                    'CentPar_ISImean', 'Temporal_dwtcArmsmean', 'FrontCent_dwtcArmsmed',
                                    'FrontCent_dwtcArmsmean', 'Occipital_dwtcDrmsmed',
                                    'Occipital_dwtcDrmsmean', 'Temporal_dwtcAvarmed',
                                    'Temporal_dwtcAvarmean', 'Temporal_dwtcDvarmed', 'target'])

        """


        # Checking missing values of each row if there is any
        # https://www.skytowner.com/explore/counting_number_of_rows_with_missing_value_in_pandas_dataframe
        missing_val_in_any_row = df.isna().any(axis=1).sum()
        print(missing_val_in_any_row)

        targetName = 'target'

        # Dropping target to train the model
        X = df.drop([targetName], axis=1).to_numpy()
        y = df[targetName].to_numpy()

        # Examining and replacing missing values with imputer function using median option. Other options available are mean, mode.
        imp = impute.SimpleImputer(strategy='median')
        imp.fit(X)
        X = imp.transform(X)

        # normalization
        Xscaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        yscaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        X = Xscaler.fit_transform(X)
        y = yscaler.fit_transform(y.reshape(-1, 1))

        # Partitioning data for training, test or validation
        ranSeed = 24060
        X_train, X_test, y_train, y_test = modelsel.train_test_split(X, y, test_size=0.3, random_state=ranSeed)





        # Training model and measuring performance with default max_depth and criterion as entropy
        #clf = andomForestClassifier (criterion='gini', splitter='best', max_depth=None, min_samples_split=2,
        #                                  min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
        #                                  random_state=ranSeed, max_leaf_nodes=None, min_impurity_decrease=0.0,
        #                                  class_weight=None, ccp_alpha=0.0)

        """
        # https://www.analyticsvidhya.com/blog/2021/06/understanding-random-forest/
        # Parameter tuning part
        clf = RandomForestClassifier(random_state=42, n_jobs=-1)
        params = {
            'criterion': ['entropy', 'gini'],
            'max_depth': [5, 10, 15],
            'min_samples_leaf': [5, 10, 20, 50, 100],
            'n_estimators': [10, 25, 30, 50, 100]
        }

        grid_search = GridSearchCV(estimator =clf,
                                   param_grid=params,
                                   cv=4,
                                   n_jobs=-1, verbose=1, scoring="accuracy")

        grid_search.fit(X_train, y_train)
        rf_best = grid_search.best_params_
        print('Best Estimator', rf_best)

        """

        criterionVal = ['entropy']
        depthVal = [5, 10, 15, 20]
        
        for ctrn in criterionVal:
            for depth in depthVal:
                clf = RandomForestClassifier(criterion=ctrn, n_estimators=100, max_depth=depth, min_samples_split=5, random_state=ranSeed)
                fit = clf.fit(X_train, y_train)
                # Measuring performance
                score = modelsel.cross_val_score(clf, X_test, y_test, cv=10)
                print("Results for criterion =", ctrn, "and max_depth =", depth, ":")
                print("Mean cross validation accuracy in array format is\n", score)
                print("Mean cross validation accuracy in percentage is", 100 * score.mean(), '%\n')
                # Confusion matrix
                predict = fit.predict(X_test)  # or y_tesy?
                print("Confusion matrix is\n", metrics.confusion_matrix(y_test, predict))
                print("--------------------------------------------------------------\n")




