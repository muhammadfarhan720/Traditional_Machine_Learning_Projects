
# Written by Muhammad Farhan Azmine

import pandas as pd
import os
import csv
import xlsxwriter
print('########## Run Started ##########')
# Data location
dirPath = '/Users/rubel/Documents/ECE5424/project/data/MIT-Dataset-Processed/'
dir_list = os.listdir(dirPath)

from glob import glob
EXT = "*.csv"
all_csv_files = [file
                 for path, subdir, files in os.walk(dirPath)
                 for file in glob(os.path.join(path, EXT))]

header = ['FIle Location', 'Channel Number', 'Channel Names']
with open('cnl_num_chk.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(header)

for csv_file in all_csv_files:
    df = pd.read_csv(csv_file)
    x = df.columns
    chnl_name = x.tolist()
    channels_num = len(df.columns)
    str_chnl_num = str(channels_num)
    str_csv_file = str(csv_file)
    str_chnl_name = str(chnl_name)
    lst_chnl_num=[]
    lst_csv_file_loc = []
    lst_chnl_name = []
    lst_chnl_num.append(str_chnl_num)
    lst_csv_file_loc.append(str_csv_file)
    lst_chnl_name.append(str_chnl_name)
    com_lst = lst_csv_file_loc + lst_chnl_num + lst_chnl_name
    with open('cnl_num_chk.csv', 'a') as f:
        write = csv.writer(f)
        write.writerow(com_lst)

print('########## Run Complete ##########')
