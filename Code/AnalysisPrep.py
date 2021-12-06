#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 07:59:14 2021

@author: jschulberg & rkelley

This script is meant for analytical purposes
"""

#%%
import pandas as pd
import numpy as np
import os
import re


try:
    adopts = pd.read_csv("../Data/Cleaned_Adoption_List.csv")
    returns = pd.read_csv("../Data/Master_Returns_List.csv")
except:
    adopts = pd.read_csv("Data/Cleaned_Adoption_List.csv")
    returns = pd.read_csv("Data/Master_Returns_List.csv")
    
#%% 
# Mark all dogs as returned in the returns dataset
returns['returned'] = 1

# Join our datasets together to create a returns indicator
dogs_joined = pd.merge(adopts,
                       returns[['LDAR ID #', 'Reason for Return', 'returned']],
                       left_on = 'ID',
                       right_on = 'LDAR ID #',
                       how = 'left')

dogs_joined.loc[dogs_joined['returned'].isna(), 'returned'] = 0

print(dogs_joined['returned'].value_counts(normalize = True))


#%% Apply data pre-processing steps
# Only bring in the columns we care about
dogs_selected = dogs_joined[['ID', 'multi_color', 'num_colors', 'MIX_BOOL', 
        'contains_black', 'contains_white',
       'contains_yellow', 'WEIGHT2', 'Age at Adoption (days)', 
       'is_retriever', 'is_shepherd', 'is_other_breed', 'num_behav_issues',
       'puppy_screen', 'new_this_week', 'needs_play', 'no_apartments',
       'energetic', 'shyness', 'needs_training', 
       'has_med_issues',
    'diarrhea',
    'ehrlichia',
    'uri',
    'ear_infection',
    'tapeworm',
    'general_infection',
    'demodex',
    'car_sick',
    'dog_park',
    'leg_issues',
    'anaplasmosis',
    'dental_issues',
    'weight_issues',
    'hair_loss',
    'treated_vaccinated',
       'returned']]

# Check to see how many NAs we have in each column
[print(col,
       '\n',
       dogs_selected[col].isna().value_counts(),
       '\n\n') 
 for col in dogs_selected.columns]

# Apply PCA

# Up/down-sample data