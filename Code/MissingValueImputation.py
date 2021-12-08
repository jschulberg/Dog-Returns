import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer

try:
    adopts = pd.read_csv("../Data/Cleaned_Adoption_List.csv")
    returns = pd.read_csv("../Data/Master_Returns_List.csv")
except:
    adopts = pd.read_csv("Data/Cleaned_Adoption_List.csv")
    returns = pd.read_csv("Data/Master_Returns_List.csv")

# %%
# Mark all dogs as returned in the returns dataset
returns['returned'] = 1

# Join our datasets together to create a returns indicator
dogs_joined = pd.merge(adopts,
                       returns[['LDAR ID #', 'Reason for Return', 'returned']],
                       left_on='ID',
                       right_on='LDAR ID #',
                       how='left')

dogs_joined.loc[dogs_joined['returned'].isna(), 'returned'] = 0

print(dogs_joined['returned'].value_counts(normalize=True))

# %% Apply data pre-processing steps
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




def na_imputation(dogs_selected):
    saved_vars = dogs_selected[['ID', "returned"]]

    ## scale data or not prior to imputation
    # scaler = MinMaxScaler()
    # scaler.fit(dogs_selected.drop(["ID", "returned"], axis = 1))
    # scaled_values = scaler.transform(dogs_selected.drop(["ID", "returned"], axis = 1))
    # data = pd.DataFrame(scaled_values, index = dogs_selected.index, columns = dogs_selected.drop(['ID', "returned"], axis = 1).columns)
    # data[['ID', "returned"]] = saved_vars
    data = dogs_selected.copy(deep=True)

    # set ID column as index
    id = data[["ID"]]
    data = data.set_index('ID')
    col_names = data.columns

    ## fill in the missing values with KNN imputer

    imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
    imputer.fit(data)

    impute1 = imputer.transform(data[:3000])
    impute2 = imputer.transform(data[3001:6000])
    impute3 = imputer.transform(data[6001:9000])
    impute4 = imputer.transform(data[9001:])

    full_impute = np.concatenate((impute1, impute2, impute3, impute4))

    dogs_full = pd.DataFrame(full_impute, columns = col_names)
    dogs_full["ID"] = id
    dogs_full = dogs_full.set_index('ID')

    return dogs_full