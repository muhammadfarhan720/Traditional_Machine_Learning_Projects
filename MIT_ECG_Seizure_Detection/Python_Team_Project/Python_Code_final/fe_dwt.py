"""
Jeremy Decker
11/23/2022
This script contains all the functions necessary to grab the DWT features from the dataset.
It will read in both folders from both datasets in order to conduct this analysis.
This code uses documentation from python libraries and from the following source:

Features Extracted from DWT:

"""
# Import Block
import pandas as pd
import numpy as np
import scipy.stats as stats
import pywt
import os
from tkinter import filedialog
import re
import scipy
from collections import Counter, defaultdict  # Unsure if this one is used.


def main():
    print("Starting DWT Analysis for a single database")
    # Fetch Folder to Analyze
    print("Please Select a Database to analyze")
    foldpath = filedialog.askdirectory()
    print("Importing Channel Map")
    # Check to see what channel map to import
    if re.search('MIT', foldpath):
        print("MIT Database Selected")
        chan_map = pd.read_excel("channel_maps_new.xlsx", sheet_name='MIT', header=0)
        base_name = "MIT_RT"
        ds_name = "MIT"
    else:
        print("Siena Database Selected")
        chan_map = pd.read_excel("channel_maps_new.xlsx", sheet_name='Siena', header=0)
        base_name = "Siena_RT"
        ds_name = "Siena"
    channel_names = chan_map.chan_name  # Need to make sure this does what I want to do, may need to convert somewhere else
    #print(channel_names)
    # Get the feature names
    feat_names = []
    features = ["dwtcAvar",  "dwtcAent", "dwtcArms", "dwtcDvar", "dwtcDent", "dwtcDrms"]
    for j in range(0, len(features)):
        for i in range(0, len(channel_names)):
            feat_name = channel_names[i] + "_" + features[j]
            feat_names.append(feat_name)
    feat_names.append("Class")
    #print("Feature Names" + str(feat_names))
    # print("Feature Names: " + str(len(feat_names)))
    # Collect all folders in the parent directory
    folds = os.listdir(foldpath)
    folds.sort(reverse=True)  # Set it to run through Seizures, then Non-Seizures, for consistency
    # Start the loop for the dwt analysis.
    dwt_extracted = []
    files_list = []
    for fold in folds:
        print("Analyzing Folder: " + fold)
        # Start looping through the folder
        fpath = str(foldpath + '/' + fold)
        # Get a list of all txt files in the folder
        print("Collecting Files")
        files = os.listdir(fpath)
        files_list.append(files)
        print("Total Files: " + str(len(files)))
        i = 1
        for file in files:
            if i % 50 == 0:
                print("Processing File: " + str(i))
            i = i+1
            # Read data into a dataframe
            filepath = fpath + '/' + file
            data = pd.read_table(filepath, header=0, sep=',')
            #print(np.shape(data))
            # Siena has a unique problem with the channel names, so check for them each time.
            if ds_name == "Siena":
                #print("Siena Database Selected")
                chan_locs = []
                chan_names = []
                for k in range(0, len(channel_names)):
                    matched = 0

                    for j in range(0, len(data.columns)):

                        test = re.match(channel_names[k], data.columns[j], flags=re.IGNORECASE)

                        if test:
                            if matched == 0:
                                chan_locs.append(j)
                                chan_names.append(data.columns[j])
                                matched = 1
                p_data = data[chan_names]
            else:
                # Run a loop to find the channel_set
                p_data = data[channel_names]
            # Run the data through the dwt_feats
            feats = dwt_feats(p_data)
            #print(np.shape(feats))
            # Add the class to the dataframe
            if fold == "NS_Segs":
                feats.append(0)
            else:
                feats.append(1)
            # print(np.shape(feats))
            dwt_extracted.append(feats)
            # print(np.shape(dwt_extracted))
            # It's putting the features into two separate rows, instead of one continuous column.
    # Convert the extracted features into a pandas dataframe
    extracted_feats = pd.DataFrame(dwt_extracted, columns=feat_names)
    # Format the column names properly.
    extracted_feats.columns = extracted_feats.columns.str.lower()
    # Add the filenames to the dataframe
    #extracted_feats['filenames'] = files_list
    # Save this as a completed dataframe
    ex_name = "dwt_feats_" + base_name + ".xlsx"
    extracted_feats.to_excel(ex_name, index=False)


def dwt_feats(pdata):
    """
    This function will take in a raw data file and extract discrete wavelet transform features
    :param pdata: 5-second segment, seizure or non-seizure
    :return results: A 1xnumchannels vector with all calculated features from the DWT
    """
    # First, get the single level dwt of the data, across the second axis of the data
    coeffs = pywt.dwt(pdata, axis=0, wavelet='sym5')
    #print("Columns of dwt:" + str(np.size(coeffs, 1)))
    #print("Rows of dwt: " + str(np.size(coeffs, 0)))
    # Get the statistics for the data, return the vector of features.
    #print("Getting Statistics for the data")
    # Do statistics 1, across axis 1 again, as it should be consistent
    # Need to fix something here, not entirely sure how. Think on this one.
    features_list = []
    for coeff in coeffs:
        #print(coeff[1:10, :])
        var = np.var(coeff, axis=0)
        #print(np.shape(var))
        entropy = []
        for i in range(0, np.size(coeff, 1)):
            channel_coeff = coeff[:, i]
            binned_channel = get_binned_data(100, channel_coeff)
            counter_values = Counter(binned_channel).most_common()
            probabilities = [elem[1]/np.size(coeff, 0) for elem in counter_values]
            #prob_look = pd.DataFrame(data=probabilities)
            #print(prob_look.head())
            entropy_col = stats.entropy(probabilities)
            entropy.append(entropy_col)
        entropy = np.array(entropy)
        #print(np.shape(np.array(entropy)))
        rms = np.nanmean(np.sqrt(coeff**2), axis=0)
        #print(np.shape(rms))
        combined_feats = np.append(var, entropy)
        combined_feats = np.append(combined_feats, rms)
        features_list.extend(combined_feats)
        #print("Shape of the features list: " + str(np.shape(features_list)))
    return features_list


def get_binned_data(nbins, data):
    """
    This script, inspired by the one written at the website cited below, will bin a single column of data based on the
    minimum and maximum of the data, and the number of bins you provide it with.
    https://python-course.eu/numerical-programming/binning-in-python-and-pandas.php
    :param nbins: The number of bins the data wil be evenly split into.
    :param data: The single column data you wish to bin
    :return binned_data:
    """
    # I actually think that normalization might hurt things, now need to optimize a number of bins to make this work
    # Better for everything, I think.
    data_max = np.max(data)
    data_min = np.min(data)
    #print("Max: " + str(data_max) +"Min: " + str(data_min))
    bin_width = abs((data_max-data_min)/nbins)
    #print(bin_width)
    low_bins = []
    high_bins = []
    # Generate bin bounds
    for i in range(0, nbins):
        low_bound = data_min + i*bin_width
        low_bins.append(low_bound)
        high_bound = data_min + (i+1)*bin_width
        high_bins.append(high_bound)
    #print(low_bins)
    #print(high_bins)
    # Once bin bounds are made, create categorical variables based on where the data would fall.
    binned_data = []
    for i in range(0, np.size(data, 0)):
        for j in range(0, len(low_bins)):
            if low_bins[j] <= data[i] <= high_bins[j]:
                binned_data.append(j)  # Gives a relative class number based on the bin the data falls in.
    # With this, the function should be complete, and the binned data can be returned for calculation.
    #print(np.shape(binned_data))
    return binned_data


if __name__ == '__main__':
    main()
