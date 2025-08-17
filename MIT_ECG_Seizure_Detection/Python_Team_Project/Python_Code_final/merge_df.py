import pandas as pd
import csv
import os
from glob import glob
"""
# MIT Data
df1 = pd.read_csv('/Users/rubel/Documents/ECE5424/project/python_code/hjroth_S_NS_MIT.csv')
df2 = pd.read_csv('/Users/rubel/Documents/ECE5424/project/python_code/dwt_feats_MIT.csv')
df3 = pd.read_csv('/Users/rubel/Documents/ECE5424/project/python_code/Interspike_MIT.csv')
df4 = pd.read_csv('/Users/rubel/Documents/ECE5424/project/python_code/ZC_MIT.csv')
df5 = pd.read_csv('/Users/rubel/Documents/ECE5424/project/python_code/Average_MIT.csv')
df6 = pd.read_csv('/Users/rubel/Documents/ECE5424/project/python_code/chnl_band_power_MIT.csv')

result = pd.concat([df1, df2,df3, df4,df5, df6], axis=1, join='inner')
merged_csv = result.to_csv("final_database_MIT.csv", index=False)

# Siena Data
df7 = pd.read_csv('/Users/rubel/Documents/ECE5424/project/python_code/hjroth_S_NS_Siena.csv')
df8 = pd.read_csv('/Users/rubel/Documents/ECE5424/project/python_code/Interspike_Siena.csv')
df9 = pd.read_csv('/Users/rubel/Documents/ECE5424/project/python_code/ZC_Siena.csv')
df10 = pd.read_csv('/Users/rubel/Documents/ECE5424/project/python_code/dwt_feats_Siena.csv')
df11 = pd.read_csv('/Users/rubel/Documents/ECE5424/project/python_code/Average_Siena.csv')
df12 = pd.read_csv('/Users/rubel/Documents/ECE5424/project/python_code/chnl_band_power_Siena.csv')

result = pd.concat([df7, df8, df9,df10, df11, df12], axis=1, join='inner')
merged_csv = result.to_csv("final_database_Siena.csv", index=False)
"""

# RT Data MIT
df13 = pd.read_csv('/Users/rubel/Documents/ECE5424/project/python_code/RT_hjroth_S_NS_MIT.csv')
df14 = pd.read_csv('/Users/rubel/Documents/ECE5424/project/python_code/RT_chnl_band_power_MIT.csv')
df15 = pd.read_csv('/Users/rubel/Documents/ECE5424/project/python_code/Interspike_MIT_RT.csv')
df16 = pd.read_csv('/Users/rubel/Documents/ECE5424/project/python_code/dwt_feats_MIT_RT.csv')
df17 = pd.read_csv('/Users/rubel/Documents/ECE5424/project/python_code/RT_ZC_MIT_Final.csv')
df18 = pd.read_csv('/Users/rubel/Documents/ECE5424/project/python_code/RT_Average_MIT_Final.csv')

result = pd.concat([df13, df14, df15, df16, df17, df18], axis=1, join='inner')
merged_csv = result.to_csv("RT_MIT1.csv", index=False)

"""
# RT Data Siena
df19 = pd.read_csv('/Users/rubel/Documents/ECE5424/project/python_code/RT_hjroth_S_NS_Siena.csv')
df20 = pd.read_csv('/Users/rubel/Documents/ECE5424/project/python_code/RT_chnl_band_power_Siena.csv')
df21 = pd.read_csv('/Users/rubel/Documents/ECE5424/project/python_code/RT_Interspike_Siena.csv')
df22 = pd.read_csv('/Users/rubel/Documents/ECE5424/project/python_code/dwt_feats_Siena_RT.csv')
df23 = pd.read_csv('/Users/rubel/Documents/ECE5424/project/python_code/RT_ZC_Siena.csv')
df24 = pd.read_csv('/Users/rubel/Documents/ECE5424/project/python_code/RT_Average_Siena_Final.csv')

result = pd.concat([df19, df20, df21,df22, df23, df24], axis=1, join='inner')
merged_csv = result.to_csv("RT_Siena.csv", index=False)

"""