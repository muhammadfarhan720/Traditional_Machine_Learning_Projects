"""
Jeremy Decker
11/25/2022
This script will take in the overall dataset file, and append grouped features for each group to it, based on a channel
map loaded at the beginning of the script.
The script uses documentation from the various libraries utilized in order to calculate its features, and save
"""

# Import block
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import *
from tkinter import filedialog
import re
import os
from ismember import ismember
# Add more as needed


def main():
    print("Starting grouped analysis")
    print("Please choose the database to collect grouped features from")
    root = Tk()
    root.filename = filedialog.askopenfilename()
    print("Importing Channel Map")
    # Match the Regular expression to the folder map in order to identify which database is needed.
    if re.search('MIT', root.filename):
        print("MIT Database Selected")
        chan_map = pd.read_excel("channel_maps_new.xlsx", sheet_name='MIT', header=0)
        database_name = "MIT_RT"
    else:
        print("Siena Database Selected")
        chan_map = pd.read_excel("channel_maps_new.xlsx", sheet_name='Siena', header=0)
        database_name = "Siena_RT"
    data = pd.read_table(root.filename, sep=',', header=0)
    root.destroy()

    # Features are consistent across each database, so load in the common sheet in order to have that list ready.
    print("Reading in Features list")
    feats = pd.read_excel("Features_List_Full.xlsx", header=0)
    print("Collecting All Channel Locations")
    chan_locs = []
    chan_strings = []
    for i in range(0, len(chan_map.locs)):
        if i == 0:
            chan_locs.append(chan_map.locs[i])
            chan_strings = chan_map.locs[i]
        else:
            test_string = re.search(chan_map.locs[i], chan_strings, flags=re.IGNORECASE)
            if not test_string:
                chan_locs.append(chan_map.locs[i])
                chan_strings +=chan_map.locs[i]

    print(chan_locs)
    # Create a loop to grab all unique channel locations
    print("Starting Loop now")
    # Start a loop to run through each feature
    for i in range(0, len(feats.feats)):
        print("Current Feature: " + feats.feats[i])
        # Secondary loop to run through each location
        for j in range(0, len(chan_locs)):
            # Tertiary loop to run through all channels, to match to location, to add to a vector for processing.
            grouped_inds = []
            grouped_channels = []
            for k in range(0, len(data.columns)):
                feature_test = re.search(feats.feats[i], data.columns[k], flags=re.IGNORECASE)
                #print("Feature Test: " + str(feature_test))
                if feature_test:
                    # Loop to check if the channel map matches the data columns.
                    for m in range(0, len(chan_map.locs)):
                        # Check to see if the location on the map is the right one
                        if re.match(chan_locs[j], chan_map.locs[m], flags=re.IGNORECASE):
                            # Check to see if the channel identified is the right one
                            name_test = "^" + chan_map.chan_name[m]  # Modify channel name to avoid issues with SPO2 or similar
                            chan_test = re.search(name_test, data.columns[k], flags=re.IGNORECASE)
                            #print("Channel Test: " + str(chan_test))
                            if chan_test:  # re.search(chan_map.chan_name[m], data.columns[k], flags=re.IGNORECASE):
                                #print("Channel Match found")
                                grouped_inds.append(k)
                                grouped_channels.append(data.columns[k])
            # After going through all data locations, process the data that you have.
            print("All channels found, processing feature " + feats.feats[i] + "Location " + chan_locs[j])
            print("Channels Used: ", str(grouped_inds))
            print("Channel Names: ", str(grouped_channels))
            grouped_set = data[grouped_channels].to_numpy()  #data.iloc[grouped_inds].to_numpy()  # Double check to make sure this does what you want it to
            print("Grouped Subset Shape: " + str(np.shape(grouped_set)))
            # Get the median values rowwise, not columnwise, to get the values we want.
            feat_med = np.nanmedian(grouped_set, axis=1)
            feat_mean = np.nanmean(grouped_set, axis=1)
            # Add these values to an pandas dataframe to be added to the main one at the end.
            feat_name = chan_locs[j] + "_" + feats.feats[i]
            feat_namemed = feat_name + "med"
            feat_namemean = feat_name + "mean"
            if i == 0:
                print("First feature, first location, defining new dataframe")
                grouped_feats = pd.DataFrame(feat_med, columns=[feat_namemed])
                grouped_feats[feat_namemean] = feat_mean
            else:
                print("Adding to new dataframe")
                grouped_feats[feat_namemed] = feat_med
                grouped_feats[feat_namemean] = feat_mean

    print("Processing Complete, appending grouped features to single channel features")
    combined_dataset = pd.concat([data, grouped_feats], axis=1)
    print("Saving File")
    # Save the appended file under a slightly modified name, so that if something is screwy, we'll be able to check.
    combined_name = "grouped_complete_" + database_name + ".xlsx"
    combined_dataset.to_excel(combined_name)
    grouped_feats.to_excel("Grouped_features.xlsx")


if __name__ == '__main__':
    main()