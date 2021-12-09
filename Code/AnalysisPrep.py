#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 07:59:14 2021

@author: jschulberg & rkelley

This script is meant for analytical purposes. It merges the adoptions and returns
datasets together on Dog ID # and creates an indicator field `returned` for
the dogs with successful matches (~8% of the total).

So far it includes:
    
    - PCA
    
PCA
PCA was attempted on the continuous and binary features in the main adoptions
dataset. The results in this section leave a lot to be desired. Performing PCA
doesn't particularly help slim down the feature space and the results aren't
particularly meaningful.

"""

#%%
import pandas as pd
import numpy as np
import os
import re # Used for Regex
from sklearn.decomposition import PCA # Used for principal component analysis
from sklearn.preprocessing import StandardScaler # Used for scaling data
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d # Used for 3D plots
from imblearn.over_sampling import ADASYN, SMOTE # Used for upsampling data
from imblearn.under_sampling import RandomUnderSampler # Used for downsampling data
from imblearn.pipeline import Pipeline # Used for building a resampling pipeline
from collections import Counter # Used for counting distributions of data


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
dogs_selected = dogs_joined[['ID', 'SEX_Male', 'SEX_Female', 'multi_color', 'num_colors', 'MIX_BOOL',
        'contains_black', 'contains_white',
       'contains_yellow', 'WEIGHT2', 'Age at Adoption (days)', 
       'is_retriever', 'is_shepherd', 'is_other_breed', 'num_behav_issues',
       'puppy_screen', 'new_this_week', 'needs_play', 'no_apartments',
       'energetic', 'shyness', 'needs_training', 
       'BULLY_SCREEN',
       'BULLY_WARNING',
       'OTHER_WARNING',
       'CATS_LIVED_WITH',
       'CATS_TEST',
       'KIDS_FIXED',
       'DOGS_IN_HOME',
       'DOGS_REQ',
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
    'HW_FIXED', 
    'FT_FIXED',
    'spay_neuter',
       'returned']]

# Check to see how many NAs we have in each column
[print(col,
       '\n',
       dogs_selected[col].isna().value_counts(),
       '\n\n') 
 for col in dogs_selected.columns]


#%% Apply PCA

# Separate out the features and replace NA's with the median of a given column
# Mean wouldn't make sense for our boolean columns
X = dogs_selected.fillna(dogs_selected.median()) \
                    .drop(columns = ['ID', 'returned']) \
                    .values

# Separate out the target
y = dogs_selected.loc[:, ['returned']].values

# Standardize the features since PCA is so heavily affected by scale
X = StandardScaler().fit_transform(X)

# Apply PCA with two components
pca = PCA().fit(X)

# Let's see how much gets explained by PCA
print(pca.explained_variance_ratio_)

# Plot the cumultaive variance explained by each component
plt.plot(np.cumsum(pca.explained_variance_ratio_ * 100),
         color = 'slateblue',
         linewidth = 3)
plt.xlabel('Number of Components', fontsize = 14)
plt.ylabel('Cumulative Variance\nExplained (%)', fontsize = 14)
plt.title('% Variation Explained by PCA', fontsize = 18)

try:
    plt.savefig('Images/% Variation Explained by PCA.png', bbox_inches='tight')

except:
    plt.savefig('../Images/% Variation Explained by PCA.png', bbox_inches='tight')

plt.show()

# Yeesh this is pretty rough. Looking at this graph, we see that 
# the data is pretty evenly described by each of the principal components.
# We can see this in the almost linear shape of the graph. That is, for each
# successive principal component included, about the same amount of variance
# gets explained.


#%% Let's try to plot the first two components
pca = PCA().fit_transform(X)

pca_df = pd.DataFrame(data = pca[:, :3], 
                           columns = ['pca_1', 'pca_2', 'pca_3'])

# Bring in the returned column from earlier
pca_df = pd.concat([pca_df, dogs_selected[['returned']]], axis = 1)

#%% Plot our results in 2-D
plt.figure(figsize = (8,8))

scatter_plot = plt.scatter(pca_df['pca_1'], 
            pca_df['pca_2'], 
            c = pca_df['returned'],
            cmap = 'Purples',
            edgecolor = 'Gray',
            alpha = .6)
plt.xlabel('Principal Component 1', fontsize = 14)
plt.ylabel('Principal Component 2', fontsize = 14)
plt.title('2 Component PCA', fontsize = 18)

# Set legend (0 = Not Returned; 1 = Returned)
plt.legend(*scatter_plot.legend_elements())

try:
    plt.savefig('Images/Top 2 Principal Components.png', bbox_inches='tight')

except:
    plt.savefig('../Images/Top 2 Principal Components.png', bbox_inches='tight')

plt.show()


#%% Let's try a 3-D Representation of the data

# Creating figure with a 3D projection
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
 
# Create a 3D scatter plot using the first three principal components
ax.scatter3D(pca_df['pca_1'], 
            pca_df['pca_2'],
            pca_df['pca_3'], 
            c = pca_df['returned'],
            s = 35,
            cmap = 'Purples',
            edgecolor = 'Gray',
            alpha = .7)
ax.set_xlabel('Principal Component 1', fontsize = 12)
ax.set_ylabel('Principal Component 2', fontsize = 12)
ax.set_zlabel('Principal Component 3', fontsize = 12)
plt.title('3 Component PCA', fontsize = 18)

# Set legend (0 = Not Returned; 1 = Returned)
plt.legend(*scatter_plot.legend_elements())

try:
    plt.savefig('Images/Top 3 Principal Components.png', bbox_inches='tight')

except:
    plt.savefig('../Images/Top 3 Principal Components.png', bbox_inches='tight')

plt.show()


#%% Up/down-sample data
# Before resampling our data, let's keep track of our count of returned vs. 
# not returned
counter = Counter(dogs_selected['returned'])
print("Distribution of Returned vs. Not Returned:\n", counter)

def resample_data(X, y):
    
    # Oversample the minority class to have 30% the number of data points
    # as the original majority class. We'll use the Adaptive synthetic 
    # sampling approach (ADASYN), which uses a density distribution to
    # generate the new samples
    over = ADASYN(sampling_strategy = .3)
    # Undersample the majority class to have 50% more data points than the
    # new minority class
    under = RandomUnderSampler(sampling_strategy = .5)
    # Chain the resampling steps into a pipeline
    sampling_steps = [('o', over), ('u', under)]
    samp_pipeline = Pipeline(steps = sampling_steps)

    X_resampled, y_resampled = samp_pipeline.fit_resample(X, y)
    
    # Let's see the counter of the distribution of our new `y_resampled`
    counter = Counter(y_resampled)
    print("Resampled Distribution of Returned vs. Not Returned:\n", counter)    
    
    return X_resampled, y_resampled

# resample our data
X_resampled, y_resampled = resample_data(X, y)

# Reapply PCA and plot
pca_resampled = PCA().fit_transform(X_resampled)

pca_resampled_df = pd.DataFrame(data = pca_resampled[:, :3], 
                           columns = ['pca_resampled_1', 
                                      'pca_resampled_2', 
                                      'pca_resampled_3'])

# Bring in the returned column from earlier
pca_resampled_df = pd.concat([pca_resampled_df, 
                              pd.DataFrame(data = y_resampled, 
                                           columns = ['returned'])], axis = 1)

plt.figure(figsize = (8,8))

scatter_plot = plt.scatter(pca_resampled_df['pca_resampled_1'], 
            pca_resampled_df['pca_resampled_2'], 
            c = pca_resampled_df['returned'],
            cmap = 'Purples',
            edgecolor = 'Gray',
            alpha = .25)
plt.xlabel('Principal Component 1', fontsize = 14)
plt.ylabel('Principal Component 2', fontsize = 14)
plt.title('2 Component PCA on Oversampled\nData Using ADASYN', fontsize = 18)

# Set legend (0 = Not Returned; 1 = Returned)
plt.legend(*scatter_plot.legend_elements())

try:
    plt.savefig('Images/Resampled - Top 2 Principal Components.png', bbox_inches='tight')

except:
    plt.savefig('../Images/Resampled - Top 2 Principal Components.png', bbox_inches='tight')

plt.show()


# Creating figure with a 3D projection
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
 
# Create a 3D scatter plot using the first three principal components
ax.scatter3D(pca_resampled_df['pca_resampled_1'], 
            pca_resampled_df['pca_resampled_2'],
            pca_resampled_df['pca_resampled_3'], 
            c = pca_resampled_df['returned'],
            s = 35,
            cmap = 'Purples',
            edgecolor = 'Gray',
            alpha = .4)
ax.set_xlabel('Principal Component 1', fontsize = 12)
ax.set_ylabel('Principal Component 2', fontsize = 12)
ax.set_zlabel('Principal Component 3', fontsize = 12)
plt.title('3 Component PCA on Oversampled\nData Using ADASYN', fontsize = 18)

# Set legend (0 = Not Returned; 1 = Returned)
plt.legend(*scatter_plot.legend_elements())

try:
    plt.savefig('Images/Resampled - Top 3 Principal Components.png', bbox_inches='tight')

except:
    plt.savefig('../Images/Resampled - Top 3 Principal Components.png', bbox_inches='tight')

plt.show()

