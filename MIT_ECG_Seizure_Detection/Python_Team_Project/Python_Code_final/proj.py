import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.datasets import make_classification
import functions
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
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
import graphviz
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import pydotplus
import collections
import pandas as pd
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import linear_model as linmod
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import feature_selection as fs
from sklearn import feature_selection as sfs
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import impute
from sklearn import preprocessing as skpreproc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy as np

# https://stackoverflow.com/questions/64343345/how-to-select-best-features-in-dataframe-using-the-information-gain-measure-in-s
# Reading the data
dirPath = '/Users/rubel/Documents/ECE5424/project/python_code/'
#fileName = 'grouped_complete_MIT_new.xlsx'
fileName = 'grouped_complete_Siena_new.xlsx'
#fileName = 'grouped_complete_Siena_RT.xlsx'
#fileName = 'grouped_complete_MIT_RT.xlsx'

df = pd.read_excel(dirPath + fileName)
#print('dataframe', df)
targetName = 'target'
# Parameters for MLP Classifier
hl = (10,30,10)
epochs = 1000
alpha= 0.0001
randomSeed = 24060

#calculate Information Gain
#functions.functions.calc_IG(dirPath, fileName)

# MLP Classifier
#functions.functions.mlp_clf(dirPath, fileName, hl, epochs, alpha, randomSeed)

# Decision Tree Classifier
functions.functions.decision_tree_classifier(dirPath, fileName)
