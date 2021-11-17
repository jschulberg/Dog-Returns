# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 12:12:42 2021

@author: jtschulberg
"""

#%%
import os
import pandas as pd
import numpy as np
import re

#%% Read in Data

# Keep track of our column names we care about
cols_to_keep = []

for file in os.listdir("Data"):
    if '.csv' in file and '2021' in file:
        # Read in CSV and skip first row
        dog_list_2021 = pd.read_csv('Data/' + file, header = 0, skiprows = [1])
        dog_list_2021['year'] = 2021
    if '.csv' in file and '2020' in file:
        # Read in CSV and skip first row
        dog_list_2020 = pd.read_csv('Data/' + file, header = 1, skiprows = [2])
        dog_list_2020['year'] = 2020
    elif '.xlsx' in file:
        # Read in all sheets as an ordered dictionary
        dog_list_archive_sheets = pd.read_excel('Data/' + file, sheet_name = None)
        
        dog_list_archive = pd.DataFrame()
        # Loop through each sheet and append it to our dataframe
        for sheet in dog_list_archive_sheets.keys():
            temp_df = dog_list_archive_sheets[sheet]
            
            # Pull out the year from the name
            year = re.findall('[0-9]+', sheet)
            if len(year) > 0:
                temp_df['year'] = year[0]
                
                # Append columns to keep
                cols_to_keep = [col for col in temp_df.columns if 'Unnamed' not in col]
                
                dog_list_archive = pd.concat([dog_list_archive, temp_df[cols_to_keep]], sort = False)
            

#%% Rename like columns
dog_list_2020 = dog_list_2020.rename(columns = {'ID #': 'ID'})
dog_list_2021 = dog_list_2021.rename(columns = {'GENDER': 'SEX'})
dog_list_archive = dog_list_archive.rename(columns = {'LBS': 'WEIGHT',
                                                      'PRIMARY BREED': 'BREED MIXES',
                                                      'Petfinder Link': 'LINK'})
        
#%% Concat everything together
dog_list_all = pd.concat([dog_list_2020, dog_list_2021, dog_list_archive], sort = False)


#%% Remove potentially sensitive columns related to the adopter
dog_list_dropped = dog_list_all.copy()
dog_list_dropped = dog_list_dropped.drop(columns = ['AC', 'FOSTER / BOARDING', 'FOSTER EMAIL',
                                                    'FOSTER', 'ADOPTER'])
    
#%% Return a master dog list
dog_list_dropped.to_csv('Data/Master_Adoption_List.csv', index = False)
