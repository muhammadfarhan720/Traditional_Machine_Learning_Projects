import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
import utils
import scipy
from itertools import zip_longest
import itertools
from scipy.stats import zscore
from scipy import signal
from scipy.integrate import simps
import yasa
import numpy as np
import seaborn as sns
import csv
import os
from glob import glob

print('####### Run Started ##########')

#dirPath = '/Users/rubel/Documents/ECE5424/project/data/siena/'
#dirPath = '/Users/rubel/Documents/ECE5424/project/data/MIT/'
#dirPath = '/Users/rubel/Documents/ECE5424/project/data/RT_Final_MIT/'
dirPath = '/Users/rubel/Documents/ECE5424/project/data/RT_Final_Siena/'
dir_list = os.listdir(dirPath)

EXT = "*.csv"
all_csv_files = [file
                 for path, subdir, files in os.walk(dirPath)
                 for file in glob(os.path.join(path, EXT))]

target = []
for csv_file in all_csv_files:
    print(csv_file)
    #print('type', type(csv_file))
    if (csv_file.find('_S_')!=-1):
        target.append('1')
        #print('Target = Seizure')
    else:
        target.append('0')
        #print('Target - Non-Seizure')


    # MIT Channels
    #df = pd.read_csv(csv_file, usecols = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3','F3-C3','C3-P3','P3-O1','FP2-F4','F4-C4', 'C4-P4', 'P4-O2',
    #                                      'FP2-F8','F8-T8','T8-P8','P8-O2','FZ-CZ','CZ-PZ']) # 'Fp2', 'Cz'
    #df.columns = map(str.lower, df.columns)
    #print(df.columns)

    #df = pd.read_csv(csv_file)


    # Siena Channels
    df = pd.read_csv(csv_file, usecols=['Fp1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T3', 'T5', 'Fc1', 'Fc5', 'Cp1',
                                        'Cp5', 'F9', 'Fz', 'Pz', 'F4', 'C4', 'P4', 'O2', 'F8',
                                        'T4', 'T6', 'Fc2', 'Fc6', 'Cp2', 'Cp6', 'F10', '1', '2'])

    #df = pd.read_csv(csv_file, usecols=['Fp1'])


    # In some files missing channel in siena 'Fp2' , 'Cz' 'EKG EKG' ('SPO2', 'HR', this 2 has all 0 value)
    # unavailable channels  ['P7-T7', 'T7-FT9', 'T8-P8-2', 'FT10-T8', 'FT9-FT10']
    #df = zscore(df)

    lst_chnl_lst = []
    ll_delta_pow = []

    # https://dsp.stackexchange.com/questions/45345/how-to-correctly-compute-the-eeg-frequency-bands-with-python
    for key, values in df.iteritems():

        values = values.tolist()
        fs = 256  # Sampling rate (2 Hz)
        # Add frequency option for 512 Hz
        # Get real amplitudes of FFT (only in postive frequencies)
        fft_vals = np.absolute(np.fft.rfft(values))
        # Get frequencies for amplitudes in Hz
        fft_freq = np.fft.rfftfreq(len(values), 1.0 / fs)
        # Define EEG bands
        eeg_bands = {'Delta': (0, 4),
                     'Theta': (4, 8),
                     'Alpha': (8, 12),
                     'Beta': (12, 30),
                     'Gamma': (30, 45)}

        # Take the mean of the fft amplitude for each EEG band
        eeg_band_fft = dict()
        for band in eeg_bands:
            freq_ix = np.where((fft_freq >= eeg_bands[band][0]) & (fft_freq <= eeg_bands[band][1]))[0]
            eeg_band_fft[band] = np.mean(fft_vals[freq_ix])

        # Find each band's power
        df = pd.DataFrame(columns=['band', 'val'])
        df['band'] = eeg_bands.keys()
        delta_band = key + '_' + df['band'].loc[0]
        theta_band = key + '_' + df['band'].loc[1]
        alpha_band = key + '_' + df['band'].loc[2]
        beta_band = key + '_' + df['band'].loc[3]
        gama_band = key + '_' + df['band'].loc[4]
        df['val'] = [eeg_band_fft[band] for band in eeg_bands]
        lst_delta_val = []
        lst_theta_val = []
        lst_alpha_val = []
        lst_beta_val = []
        lst_gama_val = []
        lst_delta_val.append(str(df['val'].loc[0]))
        lst_theta_val.append(str(df['val'].loc[1]))
        lst_alpha_val.append(str(df['val'].loc[2]))
        lst_beta_val.append(str(df['val'].loc[3]))
        lst_gama_val.append(str(df['val'].loc[4]))
        lst_chnl_name = []
        lst_chnl_name.append(delta_band)
        lst_chnl_name.append(theta_band)
        lst_chnl_name.append(alpha_band)
        lst_chnl_name.append(beta_band)
        lst_chnl_name.append(gama_band)

        lst_chnl_lst.append(lst_chnl_name)

        com_chnl_pow = lst_delta_val + lst_theta_val + lst_alpha_val + lst_beta_val + lst_gama_val
        ll_delta_pow.append(com_chnl_pow)

    all_chnl_lst = list(itertools.chain(*lst_chnl_lst))
    pow_all_band = list(itertools.chain(*ll_delta_pow))

    with open('chnl_band_power.csv', 'a') as f:
        write = csv.writer(f, quoting=csv.QUOTE_ALL)
        write.writerow(all_chnl_lst)
        write.writerow(pow_all_band)

print('####### Processing CSV FIle ##########')
# https://stackoverflow.com/questions/63705446/pandas-compare-header-with-rows-and-drop-duplicate-rows
df = pd.read_csv("chnl_band_power.csv")
df = df[df.iloc[:, 0] != df.columns[0]]
df = df.to_csv("chnl_band_power.csv", index=False)
print('Final Data\n', df)

print('target\n', target)
#print('zip\n', list(zip(target)))

t_col = pd.DataFrame({'target':target})
x = t_col.to_csv("target.csv", index=False)
df1 = pd.read_csv('chnl_band_power.csv')
df2 = pd.read_csv('target.csv')
output = pd.concat([df1, df2[['target']]], axis=1)
df = output.to_csv("chnl_band_power.csv", index=False)

"""
df = pd.read_csv("chnl_band_power.csv")
print('Final Data\n', df)
print('####### Run Complete ##########')
"""