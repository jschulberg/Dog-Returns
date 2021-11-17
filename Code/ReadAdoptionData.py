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
def read_in_dog_list(file, year):
    '''
    Function to read in the dog adoption list data, which has 12 different sheets (one
    for each month of the year).
    
    INPUTS:
        file | name of the file within the data folder
        year | string for the year this corresponds to
    '''
    # Read in all sheets as an ordered dictionary
    sheets = pd.read_excel('Data/' + file, sheet_name = None)
    
    df = pd.DataFrame()
    
    # Loop through each sheet and append it to our dataframe
    for sheet in sheets.keys():
        # Only read in the ones that start with the letter 'A'
        # (i.e. ADec or AJune)
        if sheet[0] ==  'A':
            temp_df = sheets[sheet]
            
            # Pull out the month from the name
            month = sheet[1:len(sheet)]
            temp_df['month'] = month 
            temp_df['year'] = year
            
            # Concatenate age and measure if age measure exists
            if 'age measure' in temp_df.columns:
                temp_df['AGE'] = temp_df['AGE'] + ' ' + temp_df['AGE MEASURE']
            
            # Append columns to keep
            cols_to_keep = [col for col in temp_df.columns if 'Unnamed' not in col]
            
            df = pd.concat([df, temp_df[cols_to_keep]], sort = False)
    
    return df

#%%
for file in os.listdir("Data"):
    if '2021' in file:
        print(f"Reading in {file}...")
        # Read in CSV and skip first row
        dog_list_2021 = read_in_dog_list(file, '2021')

    if '2020' in file:
        print(f"Reading in {file}...")
        # Read in CSV and skip first row
        dog_list_2020 = read_in_dog_list(file, '2020')
        
    elif 'Archived' in file:
        print(f"Reading in {file}...")
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
    else:
        print(f"Skipping {file}...")
            

#%% Rename like columns
dog_list_2020 = dog_list_2020.rename(columns = {'ID #': 'ID'})
dog_list_2021 = dog_list_2021.rename(columns = {'GENDER': 'SEX'})
dog_list_archive = dog_list_archive.rename(columns = {'LBS': 'WEIGHT',
                                                      'PRIMARY BREED': 'BREED MIXES',
                                                      'Petfinder Link': 'LINK'})
        
#%% Concat everything together
dog_list_all = pd.concat([dog_list_2020, dog_list_2021, dog_list_archive], sort = False)

# Coalesce columns that should be the same (i.e. 'ID #' and 'ID')
dog_list_all['ID'] = dog_list_all['ID'].combine_first(dog_list_all['ID #'])
dog_list_all['KIDS'] = dog_list_all['KIDS'].combine_first(dog_list_all['KIDS.1'])
dog_list_all['STATUS'] = dog_list_all['STATUS'].combine_first(dog_list_all['A/U'])
dog_list_all['DOG NAME'] = dog_list_all['DOG NAME'].combine_first(dog_list_all['DOG'])
dog_list_all['DOG NAME'] = dog_list_all['DOG NAME'].combine_first(dog_list_all['Dog'])
dog_list_all['DOG NAME'] = dog_list_all['DOG NAME'].combine_first(dog_list_all['u'])
dog_list_all['DOG NAME'] = dog_list_all['DOG NAME'].combine_first(dog_list_all['b'])
dog_list_all['BEHAVIORAL NOTES'] = dog_list_all['BEHAVIORAL NOTES'].combine_first(dog_list_all['BEHAVIORIAL NOTES'])
dog_list_all['BEHAVIORAL NOTES'] = dog_list_all['BEHAVIORAL NOTES'].combine_first(dog_list_all['NOTES'])
                
print(dog_list_all.year.value_counts())
print(dog_list_all.month.value_counts())

#%% Remove potentially sensitive columns related to the adopter
# and other columns we don't need
dog_list_dropped = dog_list_all.copy()
dog_list_dropped = dog_list_dropped.drop(columns = ['AC', 'FOSTER / BOARDING', 'FOSTER EMAIL',
                                                    'FOSTER', 'ADOPTER', 'AGE MEASURE',
                                                    'ID #', 'KIDS.1', 'DOG', 'Dog', 'b',
                                                    'A/U', 'u', 'NOTES', 'BEHAVIORIAL NOTES'])
    
# Remove any dogs that don't have IDs
dog_list_dropped = dog_list_dropped.loc[~(pd.isnull(dog_list_dropped['ID'])), :]

# Remove dogs that only have a date as an ID since there'll be dupliates anyways...
#dog_list_dropped = dog_list_dropped.loc[~(dog_list_dropped['ID'].str.contains('/')), :]


#%% Deduplicate based on ID
dog_list_deduped = dog_list_dropped.copy().drop_duplicates() \
                                            .drop_duplicates(subset = ['ID'], keep = 'first')
    
print(dog_list_deduped.year.value_counts())
print(dog_list_deduped.month.value_counts())

#%% Return a master dog list
dog_list_deduped.to_csv('Data/Master_Adoption_List.csv', index = False)
