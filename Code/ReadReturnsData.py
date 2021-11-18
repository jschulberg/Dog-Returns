# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 11:50:18 2021

@author: jtschulberg

This script reads in all of the dog returns data. Specifically, there are 
two files we're working with:
    
    1. Returns 2021 | A sheet of returns from 2021
    2. Returns Archive | A workbook with returns from every previous year,
                        going back to 2009
"""

#%%
import os
import pandas as pd
import numpy as np
import re

#%% Read in Data
def read_in_returns_list(file):
    '''
    Function to read in the dog adoption list data, which has 12 different sheets (one
    for each month of the year).
    
    INPUTS:
        file | Name of the file within the data folder
        year | String for the year this corresponds to (i.e. '2020')
    OUTPUTS:
        df | Singular DataFrame with all of our data for the year in a 
             consolidated format
    '''
    # Read in all sheets as an ordered dictionary
    sheets = pd.read_excel('Data/' + file, sheet_name = None)
    
    df = pd.DataFrame()
    
    # Loop through each sheet and append it to our dataframe
    for sheet in sheets.keys():
        # Only read in the ones that start with the letter 'A'
        # (i.e. ADec or AJune)
        if 'complete' in sheet.lower():
            print(f'Reading in sheet {sheet} from {file}...')
            temp_df = sheets[sheet]
            
            # Pull out the year from the name
            temp_df['year'] = re.findall('[0-9]+', sheet)[0]
            
            # Concatenate age and measure if age measure exists
            if 'age measure' in temp_df.columns:
                temp_df['AGE'] = temp_df['AGE'] + ' ' + temp_df['AGE MEASURE']
            
            # Append columns to keep
            cols_to_keep = [col for col in temp_df.columns if 'Unnamed' not in col]
            
            df = pd.concat([df, temp_df[cols_to_keep]], sort = False)
    
    return df

#%% Read in all of our completed returns and append to one master dataframe
all_returns = pd.DataFrame()

for file in os.listdir("Data"):
    if 'return' in file.lower():
        print(f"\nReading in {file}...")
        # Read in CSV and skip first row
        df = read_in_returns_list(file)
        
        # Append to master dataframe
        all_returns = pd.concat([all_returns, df], sort = False)


#%% Coalesce columns that should be the same (i.e. 'ID #' and 'ID')
all_returns_coalesced = all_returns.copy()
all_returns_coalesced['LDAR ID #'] = all_returns_coalesced['LDAR ID #'].combine_first(all_returns_coalesced['LD ID #'])

all_returns_coalesced['LDAR Name'] = all_returns_coalesced['LDAR Name'].combine_first(all_returns_coalesced['Name'])
all_returns_coalesced['LDAR Name'] = all_returns_coalesced['LDAR Name'].combine_first(all_returns_coalesced["Dog's Name"])
all_returns_coalesced['LDAR Name'] = all_returns_coalesced['LDAR Name'].combine_first(all_returns_coalesced["F"])
all_returns_coalesced['LDAR Name'] = all_returns_coalesced['LDAR Name'].combine_first(all_returns_coalesced["Name"])

all_returns_coalesced['Return Category'] = all_returns_coalesced['Return Category'].combine_first(all_returns_coalesced["Reason Category"])


all_returns_coalesced['Dog info'] = all_returns_coalesced['Dog info'].combine_first(all_returns_coalesced['Animal info'])

all_returns_coalesced['Reason for Return'] = all_returns_coalesced['Reason for Return'].combine_first(all_returns_coalesced['Return Category'])
all_returns_coalesced['Reason for Return'] = all_returns_coalesced['Reason for Return'].combine_first(all_returns_coalesced['Primary Reason'])

all_returns_coalesced['DOG/CAT'] = all_returns_coalesced['DOG/CAT'].combine_first(all_returns_coalesced['Type (Dog, Puppy, Cat, Kitten)'])
all_returns_coalesced['DOG/CAT'] = all_returns_coalesced['DOG/CAT'].combine_first(all_returns_coalesced['Dog / Cat'])

all_returns_coalesced['6 Month +'] = all_returns_coalesced['6 Month +'].combine_first(all_returns_coalesced['6MO+'])

all_returns_coalesced = all_returns_coalesced.drop(columns = ['LD ID #', 'AC', 'Returning Owner',
                                                              'Returning Adopter Email', 'Returning Adopter Phone',
                                                              'Rover ID', 'Type (Dog, Puppy, Cat, Kitten)',
                                                              'Intake: Physically in LDAR Care', 'Rover Correct', 
                                                              'Action Required', 'Animal info', "Dog's Name",
                                                              'Ret Owner Info', 'Returns Coordinator', 'Reactivated',
                                                              'DNA', 'DNA Updated', '2020 Animal', '2019 Animal',
                                                              '2018 Animal', '2017 ANIMAL', 'AC doing return', 'F',
                                                              'Tracking Sheet Sent', "Photos Rec'd", 'Dog / Cat',
                                                              'Name', 'Adoption Approval Type', 'Reason Category',
                                                              '6MO+'
                                                              ])

            
print(all_returns_coalesced.year.value_counts())
print(all_returns_coalesced.columns)


#%% Remove any rows we don't care about

# Some returns are for cats/kittens. Let's get rid of these
all_returns_filtered = all_returns_coalesced.copy()
# Create booleans to detect cat/kitten
cat_bool = all_returns_filtered['DOG/CAT'].str.lower().str.contains('cat').replace(np.nan, False)
kit_bool = all_returns_filtered['DOG/CAT'].str.lower().str.contains('kitten').replace(np.nan, False)
all_returns_filtered = all_returns_filtered.loc[~(cat_bool) & ~(kit_bool), :]

# Clean up the ID column
all_returns_filtered['LDAR ID #'] = all_returns_filtered['LDAR ID #'].str.strip()

# Drop duplicates
all_returns_filtered = all_returns_filtered.drop_duplicates(subset = ['LDAR ID #'])
                                                                      
print(all_returns_filtered['DOG/CAT'].value_counts())

#%% Write out our results
all_returns_filtered.to_csv('Data/Master_Returns_List.csv', index = False)