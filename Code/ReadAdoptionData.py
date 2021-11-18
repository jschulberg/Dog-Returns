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
    if 'DOG LIST 2020' in file:
        print(f"Reading in {file}...")
        # Read in CSV and skip first row
        dog_list_2021 = read_in_dog_list(file, '2021')

    elif 'DOG LIST 2021' in file:
        print(f"Reading in {file}...")
        # Read in CSV and skip first row
        dog_list_2020 = read_in_dog_list(file, '2020')
        
    elif 'DOG LIST 2019' in file:
        print(f"Reading in {file}...")
        # Read in CSV and skip first row
        dog_list_2019 = read_in_dog_list(file, '2019')
        
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
dog_list_2021 = dog_list_2021.rename(columns = {'GENDER': 'SEX'})
dog_list_archive = dog_list_archive.rename(columns = {'LBS': 'WEIGHT',
                                                      'PRIMARY BREED': 'BREED MIXES',
                                                      'Petfinder Link': 'LINK'})
        

#%% Concat everything together
dog_list_all = pd.concat([dog_list_2019, dog_list_2020, dog_list_2021, dog_list_archive], sort = False)

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



#%%##################################
##### Bring in DOB and Adoption Dates
#####################################
# NOTE: Adoption dates are located across two major datasets
# We only have DOB for 2020/2021 unfortunately
dob_2020 = pd.read_excel('Data/animal_export_2020.xlsx')
dob_2021 = pd.read_excel('Data/animal_export_2021.xlsx')
dob_2016_2019 = pd.read_excel('Data/animal_export_2016-2019.xlsx')
adoption_date = pd.read_excel('Data/adoption date to ID mapping.xlsx')
adoption_date_all = pd.concat(pd.read_excel('Data/Lucky Dog Adoption Spreadsheet.xlsx', sheet_name = None), ignore_index = False)


#%% Only pull out the columns we want and drop dupes
all_dobs = pd.concat([dob_2020, dob_2021, dob_2016_2019], sort = False)[['Animal ID', 'Date of Birth']] \
            .drop_duplicates() \
            .rename(columns = {'Animal ID': 'ID'})

adoption_cleaned = adoption_date.loc[adoption_date['Species'] == 'Dog', ['ID', 'Adopted On']] \
                                .drop_duplicates()


# Fix up the ID columns
all_dobs['ID'] = all_dobs['ID'].str.strip()
adoption_cleaned['ID'] = adoption_cleaned['ID'].str.strip()



#%% Join everything together
all_dates = pd.merge(all_dobs, 
                     adoption_cleaned, 
                     how = 'outer', 
                     on = 'ID')


#%%
adoption_date_all_cleaned = adoption_date_all.copy().loc[adoption_date_all['SPECIES'] != 'Cat', :]
# Coalesce columns that should be the same (i.e. 'ID #' and 'ID')
adoption_date_all_cleaned['AGE MEASURE'] = adoption_date_all_cleaned['AGE MEASURE'].combine_first(adoption_date_all_cleaned['Age Measure'])
adoption_date_all_cleaned['BREED'] = adoption_date_all_cleaned['BREED'].combine_first(adoption_date_all_cleaned['COAT PATTERN'])
adoption_date_all_cleaned['BREED'] = adoption_date_all_cleaned['BREED'].combine_first(adoption_date_all_cleaned['MIX'])
adoption_date_all_cleaned['BREED'] = adoption_date_all_cleaned['BREED'].combine_first(adoption_date_all_cleaned['PRIMARY BREED'])
adoption_date_all_cleaned['BREED'] = adoption_date_all_cleaned['BREED'].combine_first(adoption_date_all_cleaned['Primary Breed'])
adoption_date_all_cleaned['SECONDARY BREED'] = adoption_date_all_cleaned['SECONDARY BREED'].combine_first(adoption_date_all_cleaned['Secondary Breed'])
adoption_date_all_cleaned['SPAYED/NEUTERED'] = adoption_date_all_cleaned['SPAYED/NEUTERED'].combine_first(adoption_date_all_cleaned['SPAYED/\nNEUTERED'])

# Drop unneccessary columns
adoption_date_all_cleaned = adoption_date_all_cleaned.drop(columns = ['Age Measure', 'COAT PATTERN', 'MIX',
                                          'PRIMARY BREED', 'Primary Breed', 
                                          'ADOPTER FIRST NAME', 'ADOPTER LAST NAME', 
                                          'ADOPTION COORDINATOR', 'Address', 
                                          'Cell Phone', 'City', 'E-mail', 'E-mail 1',
                                          'FOSTER FIRST NAME', 'FOSTER LAST NAME',
                                          'FOSTER NAME', 'Home Phone', 'LDAR Rep',
                                          'MICROCHIP COMPANY', 'MICROCHIP ID', 
                                          'Contract in ROVER', 'Info in ROVER',
                                          'Intake Age (Months)', 'Intake Number',
                                          'SPAYED/\nNEUTERED', 'in process',
                                          'rescuegroups ID', 'scanned', 'Secondary Breed'
                                          ]) 

# Drop rows with no ID and duplicative IDs (there's only like 6, but still)
adoption_date_all_cleaned = adoption_date_all_cleaned.loc[adoption_date_all_cleaned['ID'] != np.nan] \
                                                    .drop_duplicates(subset = ['ID'])

# Convert Date Adopted to datetime
adoption_date_all_cleaned['DATE ADOPTED'] = adoption_date_all_cleaned['DATE ADOPTED'].fillna(pd.NaT).astype(np.datetime64, errors = 'ignore')
adoption_date_all_cleaned['DATE ADOPTED'] = pd.to_datetime(adoption_date_all_cleaned['DATE ADOPTED'], errors = 'coerce')

adoption_date_all_cleaned['BIRTHDATE'] = adoption_date_all_cleaned['BIRTHDATE'].fillna(pd.NaT).astype(np.datetime64, errors = 'ignore')
adoption_date_all_cleaned['BIRTHDATE'] = pd.to_datetime(adoption_date_all_cleaned['BIRTHDATE'], errors = 'coerce')
                       
adoption_date_all_cleaned['ID'] = adoption_date_all_cleaned['ID'].str.strip()

                             
#%% Merge these new adoption and birth dates back in
all_dates2 = all_dates.merge(adoption_date_all_cleaned[['ID', 'BIRTHDATE', 'DATE ADOPTED']],
                             how = 'outer',
                             on = 'ID')

# Coalesce our adoption and birth dates
all_dates2['Adopted On'] = all_dates2['Adopted On'].combine_first(all_dates2['DATE ADOPTED'])
all_dates2['Date of Birth'] = all_dates2['Date of Birth'].combine_first(all_dates2['BIRTHDATE'])
all_dates2 = all_dates2.drop(columns = ['DATE ADOPTED', 'BIRTHDATE']).drop_duplicates(subset = ['ID'])


#%% Merge DOB and Date of Adoption back into our master adoption sheet
master_dog_adoption = pd.merge(dog_list_deduped,
                               all_dates2,
                               on = 'ID',
                               how = 'left')


#%% Return a master dog list
master_dog_adoption.to_csv('Data/Master_Adoption_List.csv', index = False)
