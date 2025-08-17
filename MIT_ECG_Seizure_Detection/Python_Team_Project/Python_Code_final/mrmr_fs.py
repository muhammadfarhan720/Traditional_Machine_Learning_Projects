"""
Jeremy Decker
11/28/2022
This script will apply the MRMR feature selection algorithm, using the ensemble implementation
I used to following implementation of the MRMR algorithm.
https://github.com/smazzanti/mrmr
"""
# import block
import numpy as np
import pandas as pd
from mrmr import mrmr_classif
from tkinter import *
from tkinter import filedialog
import re


def main():
    # Import Dataset
    # Set whether or not you want the common variables or not. Rename the results files manually in this case.
    print("Please choose the database to collect grouped features from")
    root = Tk()
    root.filename = filedialog.askopenfilename()
    print("Importing Channel Map")
    # Match the Regular expression to the folder map in order to identify which database is needed.
    if re.search('MIT', root.filename):
        print("MIT Database Selected")
        chan_map = pd.read_excel("channel_maps_new.xlsx", sheet_name='MIT', header=0)
        base = 'MIT'
    else:
        print("Siena Database Selected")
        chan_map = pd.read_excel("channel_maps_new.xlsx", sheet_name='Siena', header=0)
        base = 'Siena'
    data = pd.read_excel(root.filename, header=0)
    # Now that the data is imported, separate the labels and the data itself
    labels = data['target']
    drop_list = []
    for i in range(0, len(data.columns)):
        test = re.search("Unnamed", data.columns[i])
        #print(test)
        if test:
            drop_list.append(data.columns[i])
    # Drop the indexing columns
    data.drop(columns=drop_list, inplace=True)
    data.drop(columns=['target'], inplace=True)
    # print("Data Shape: " + str(np.shape(data)))
    print(np.shape(labels))
    total_feats = len(data.columns)
    feats_25p = round(total_feats/4)
    feats_50p = round(total_feats/2)
    # Now, utilize mrmr to get the top features in the dataset
    sol_10 = mrmr_classif(data, labels, K=10)
    #print(sol_10)
    sol_20 = mrmr_classif(data, labels, K=20)
    sol_25p = mrmr_classif(data, labels, K=feats_25p)
    sol_50p = mrmr_classif(data, labels, K=feats_50p)
    # Now that this is done, save them
    sol_10 = pd.DataFrame(sol_10, columns=['Feats'])
    sol_20 = pd.DataFrame(sol_20, columns=['Feats'])
    sol_25p = pd.DataFrame(sol_25p, columns=['Feats'])
    sol_50p = pd.DataFrame(sol_50p, columns=['Feats'])
    sol_20.to_excel(str(base + "mrmr_20f.xlsx"))
    sol_10.to_excel(str(base + "mrmr_10f.xlsx"))
    sol_25p.to_excel(str(base + "mrmr_25pf.xlsx"))
    sol_50p.to_excel(str(base + "mrmr_50pf.xlsx"))


if __name__ == '__main__':
    main()
