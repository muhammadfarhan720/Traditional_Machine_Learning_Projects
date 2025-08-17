"""
Jeremy Decker
12/4/2022
SVC testing script
This script will run tests using an SVC classifier run on a dataset with a selected list of filters. The training process
was optimized by the intelex package for scikit learn in order to
https://github.com/intel/scikit-learn-intelex
Kfold cross validaiton was implemented using the following:
https://machinelearningmastery.com/how-to-configure-k-fold-cross-validation/
TODO- Add gridsearch, multi-featureset, potentially multidataset modeling.
"""
import pandas as pd
import numpy as np
import os
from tkinter import *
from tkinter import filedialog
from sklearnex import patch_sklearn
import re
patch_sklearn()
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit


def main():
    print("Starting up...")
    print("Please choose a dataset to work with")
    datroot = Tk()
    datroot.filename = filedialog.askopenfilename()
    print("Please choose a folder containing feature sets to work with")
    froot = filedialog.askdirectory()
    file_list = os.listdir(froot)
    print("Processing Dataset")
    # Read in dataset
    data = pd.read_excel(datroot.filename, header=0)
    # Extract labels
    labels = data['target']
    # Collect the features from the feature selection file.
    for filename in file_list:
        filepath = froot + '/' + filename
        feats = pd.read_excel(filepath, header=0)
        num_feats = len(feats.Feats)
        preds = data[feats.Feats]
        print("Imputing missing values and normalizing data")
        med_impute = SimpleImputer(strategy='median')
        preds = preds.to_numpy()
        preds = med_impute.fit_transform(preds)
        scaler = MinMaxScaler()
        preds = scaler.fit_transform(preds)
        print("Creating K-folds")
        # Set up the kfolds
        cv = KFold(n_splits=5, random_state=42, shuffle=True)
        train_d, test_d, train_l, test_l = train_test_split(preds, labels, test_size=0.3,
                                                            random_state=42)
        [blah, subset_d, bleh, subset_l] = train_test_split(preds, labels, test_size=0.2,
                                                            random_state=84)
        model_rbf = SVC()
        print("Determining Optimal Gamma and C values")
        C_range = np.logspace(-6, 4, 8)
        gamma_range = np.logspace(-5, 5, 8)
        param_grid = dict(gamma=gamma_range, C=C_range)
        # Then, subsample the data into 10 splits(more than what the tutorial did) to put into the search
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.1, random_state=42)
        grid = GridSearchCV(model_rbf, param_grid=param_grid, cv=cv)
        grid.fit(subset_d, subset_l)  # 30% subsample to find ideal parameters
        print("Optimal Parameter: " + str(grid.best_params_) + " Optimal Score: " + str(grid.best_score_))
        best_params = grid.best_params_
        optgamma = best_params['gamma']
        optc = best_params['C']
        model_poly6 = SVC(kernel='poly', degree=6)
        model_poly5 = SVC(kernel='poly', degree=5)
        model_linear = SVC(kernel='linear')
        model_rbf = SVC(kernel='rbf', gamma=optgamma, C=optc)
        model_list = [model_rbf, model_poly5, model_poly6, model_linear]
        it = 0
        model_names = ['rbf', 'poly5', 'poly6', 'linear']
        for model in model_list:
            # Train a version of the model on a more traditional dataset
            print("Model = " + model_names[it])
            model_test = model
            model_test.fit(train_d, train_l)
            train_score = model_test.score(train_d, train_l)
            test_score = model_test.score(test_d, test_l)
            test_preds = model_test.predict(test_d)
            conf_mat = confusion_matrix(test_l, test_preds)


            # Run the cross validation scoring using the created kfold, and all processors in order to speed things up.
            scores = cross_val_score(model, preds, labels, cv=cv, n_jobs=-1)
            # Fetch the confusion Matrix
            #print("Mean Accuracy for model " + model_names[it])
            #print(np.mean(scores))
            if it == 0:
                score_set = pd.DataFrame(scores, columns=[model_names[it]])
                score_set[str(model_names[it] + "_SingletrainAcc")] = train_score
                score_set[str(model_names[it] + "_SingletestAcc")] = test_score
                conf_set = pd.DataFrame(np.ravel(conf_mat), columns=[str(model_names[it] + "_SingletrainAcc")])
                it += 1
            else:
                score_set[model_names[it]] = scores
                score_set[str(model_names[it] + "_SingletrainAcc")] = train_score
                score_set[str(model_names[it] + "_SingletestAcc")] = test_score
                conf_set[str(model_names[it] + "_SingleConf_mat")] = np.ravel(conf_mat)
                it += 1
        # Extract Optimal data paradigm
        score_set['opt_gamma'] = optgamma
        score_set['opt_C'] = optc
        print("Saving Results")
        if re.search('MIT', datroot.filename):
            print("MIT Database Selected")
            base = 'MIT'
        else:
            print("Siena Database Selected")
            base = 'Siena'
        res_name = base+"_" + str(num_feats) + "_" + "SVC_Results.xlsx"
        conf_name = base+"_" + str(num_feats) + "_" + "SVC_Conf_mats.xlsx"
        score_set.to_excel(res_name)
        conf_set.to_excel(conf_name)


if __name__ == '__main__':
    main()
