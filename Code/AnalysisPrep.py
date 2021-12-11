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
    - Resampling (Oversampling minority class + Undersampling majority class)
    
PCA
PCA was attempted on the continuous and binary features in the main adoptions
dataset. The results in this section leave a lot to be desired. Performing PCA
doesn't particularly help slim down the feature space and the results aren't
particularly meaningful.

Resampling
We opted to leverage the Adaptive Synthetic (ADASYN) oversampling technique on 
the ‘returned’ minority class; and a Random undersampling technique on the ‘not 
returned’ majority class. We found ADASYN, because of its ability to put synthetic
 points in low distributed areas of the data and then use K-Nearest Neighbors to 
 classify them, to give us the best representation of the minority class. However,
 we opted not to use ADASYN to get the minority class to the same size (9664) as 
 the majority class; instead opting to increase the size by 3x (2871 ‘returned’ dogs).
 To get the majority class down in size, we just used a random undersampler to
 randomly select data points from the majority class (‘not returned’) to keep, 
 ultimately ending with about half as many points (5742 ‘not returned’ dogs).

| Class	| # of Original Data Points | Resampled Data |
| --- | --- | --- |
| ‘Returned’ | 824 | 2871 |
| ‘Not Returned’ | 9664 | 5742 |


"""

#%%
import pandas as pd
import numpy as np
import os
import re # Used for Regex
from sklearn.decomposition import PCA # Used for principal component analysis
import prince
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
dogs_selected = dogs_joined[['ID', 
                             'SEX', 
                             'multi_color', 
                             'num_colors', 
                             'MIX_BOOL',
                             'contains_black', 
                             'contains_white',
                            'contains_yellow', 
                            'WEIGHT2', 
                            'Age at Adoption (days)', 
                            'is_retriever', 
                            'is_shepherd', 
                            'is_other_breed', 
                             'num_behav_issues',
                            'puppy_screen', 
                            'needs_play', 
                            'no_apartments',
                            'energetic', 
                            'shyness', 
                            'needs_training', 
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
# [print(col,
#        '\n',
#        dogs_selected[col].isna().value_counts(),
#        '\n\n') 
#  for col in dogs_selected.columns]


#%% Apply PCA

# Separate out the features and replace NA's with the median of a given column
# Mean wouldn't make sense for our boolean columns
def apply_pca_variance_explained(df):
    X = dogs_selected.fillna( df.median()) \
                        .drop(columns = ['ID', 'returned']) \
                        .values
    
    # Separate out the target
    y =  df.loc[:, ['returned']].values
    
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
    
    return X, y, pca

# X, y, pca = apply_pca_variance_explained(dogs_selected)
    


# Yeesh this is pretty rough. Looking at this graph, we see that 
# the data is pretty evenly described by each of the principal components.
# We can see this in the almost linear shape of the graph. That is, for each
# successive principal component included, about the same amount of variance
# gets explained.


#%% Let's try to plot the first two components
def apply_pca_main_components(X, plot_names):
    pca = PCA().fit_transform(X)
    
    pca_df = pd.DataFrame(data = pca[:, :3], 
                               columns = ['pca_1', 'pca_2', 'pca_3'])
    
    # Bring in the returned column from earlier
    pca_df = pd.concat([pca_df, dogs_selected[['returned']]], axis = 1)
    
    # Plot our results in 2-D
    plt.figure(figsize = (8,8))
    
    scatter_plot = plt.scatter(pca_df['pca_1'], 
                pca_df['pca_2'], 
                c = pca_df['returned'],
                cmap = 'Purples',
                edgecolor = 'Gray',
                alpha = .6)
    plt.xlabel('Principal Component 1', fontsize = 14)
    plt.ylabel('Principal Component 2', fontsize = 14)
    plt.title(plot_names[0], fontsize = 18)
    
    # Set legend (0 = Not Returned; 1 = Returned)
    plt.legend(*scatter_plot.legend_elements())
    
    try:
        plt.savefig('Images/' + plot_names[0] + '.png', bbox_inches='tight')
    
    except:
        plt.savefig('../Images/' + plot_names[0] + '.png', bbox_inches='tight')
    
    plt.show()
    
    
    # Let's try a 3-D Representation of the data
    
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
    plt.title(plot_names[1], fontsize = 18)
    
    # Set legend (0 = Not Returned; 1 = Returned)
    plt.legend(*scatter_plot.legend_elements())
    
    try:
        plt.savefig('Images/' + plot_names[1] + '.png', bbox_inches='tight')
    
    except:
        plt.savefig('../Images/' + plot_names[1] + '.png', bbox_inches='tight')
    
    plt.show()
    
    return pca_df

# pca_df = apply_pca_main_components(X, 
#                                    plot_names = ['Top 2 Principal Components', 
#                                                  'Top 3 Principal Components'])


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
# X_resampled, y_resampled = resample_data(X, y)



# resampled_pca_df = apply_pca_main_components(X_resampled, 
#                                              plot_names = ['Resampled - Top 2 Principal Components',
#                                                            'Resampled - Top 3 Principal Components'])

#%% Attempt to use Multiple Correspondence Analysis (MCA) on the resampled data
# Multiple correspondence analysis (MCA) is an extension of correspondence 
# analysis (CA). It should be used when you have more than two categorical variables. 
# The idea is simply to compute the one-hot encoded version of a dataset and apply CA on it. 
def apply_mca_and_plot(df):
    mca = prince.MCA(
        n_components = 15,
        n_iter = 50,
        copy = True,
        check_input = True,
        engine = 'auto')
    
    X_mca = dogs_selected.fillna(df.median()) \
                        .drop(columns = ['ID', 'returned']) 
                        
    X_mca, y_mca = resample_data(X_mca, y)
                        
    mca_df = mca.fit(X_mca).transform(X_mca)
    # Rename the columns by adding 'mca_' to each
    mca_df.columns = ['mca_' + str(col) for col in mca_df.columns]
    mca_df = pd.concat([mca_df, df['returned']], axis = 1)
    
    # Plot the results
    plt.figure(figsize = (8,8))
    
    scatter_plot = plt.scatter(mca_df['mca_0'], 
                mca_df['mca_1'], 
                c = mca_df['returned'],
                cmap = 'Purples',
                edgecolor = 'Gray',
                alpha = .4)
    plt.xlabel('Component 1', fontsize = 14)
    plt.ylabel('Component 2', fontsize = 14)
    plt.title('2 Component MCA on Oversampled\nData Using ADASYN', fontsize = 18)
    
    # Set legend (0 = Not Returned; 1 = Returned)
    plt.legend(*scatter_plot.legend_elements())
    
    plt.savefig('Images/Resampled - Top 2 MCA Components.png', bbox_inches='tight')
    plt.show()
    
    
    # Creating figure with a 3D projection
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
     
    # Create a 3D scatter plot using the first three principal components
    ax.scatter3D(mca_df['mca_0'], 
                mca_df['mca_1'],
                mca_df['mca_2'], 
                c = mca_df['returned'],
                s = 35,
                cmap = 'Purples',
                edgecolor = 'Gray',
                alpha = .5)
    ax.set_xlabel('Component 1', fontsize = 12)
    ax.set_ylabel('Component 2', fontsize = 12)
    ax.set_zlabel('Component 3', fontsize = 12)
    plt.title('3 Component MCA on Oversampled\nData Using ADASYN', fontsize = 18)
    
    # Set legend (0 = Not Returned; 1 = Returned)
    plt.legend(*scatter_plot.legend_elements())
    
    plt.savefig('Images/Resampled - Top 3 MCA Components.png', bbox_inches='tight')
    plt.show()
    
    return mca_df

# mca_df = apply_mca_and_plot(dogs_selected)


#%% Attempt Factor Analysis of Mixed Data (FAMD)
# Start by converting our 0's and 1's to 'No' and 'Yes' respectively

# famd = prince.FAMD(n_components = 2, 
#                    n_iter = 3)


# famd_df = famd.fit(X_mca).transform(X_mca)
