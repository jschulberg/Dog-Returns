
#%%
import pandas as pd
import numpy as np
import os
import re
from itertools import product

# It looks like Spyder and PyCharm have slightly different ways of reading
# in data, so I'm including a try/except block so we don't have to change code.
try:
    adopts = pd.read_csv("../Data/Master_Adoption_List.csv")
    returns = pd.read_csv("../Data/Master_Returns_List.csv")
except:
    adopts = pd.read_csv("Data/Master_Adoption_List.csv")
    returns = pd.read_csv("Data/Master_Returns_List.csv")


# %% Clean up color column
def clean_color(df):
    '''
    This function is meant to clean up the COLOR column, which denotes the color
    of a given dog. Because there are so many combinations of different colors for
    dogs, the main purpose is to create multiple indicator columns for whether or
    not a dog has a certain color in it (black, yellow, white, etc.) and whether
    the dog is a solid color or multi-colored.

    After cleaning up the `COLOR` column with `COLOR_FIXED`, it creates the following features:
        - `multi_color` | Variable indicating whether  or not a dog has multiple colors denoted
                            in it. This is calculated by seeing if the number of commas (',')
                            in the `COLOR_FIXED` column is greater than 1 or
                            whether the color is denoted as 'tri-color'.
        - `num_colors` | Continuous variable counting up the number of commas (',')
                            in the `COLOR_FIXED` column.
        - `contains_black` | Variable indicating whether 'black' appears in the
                                `COLOR_FIXED` column.
        - `contains_white` | Variable indicating whether 'white' appears in the
                                `COLOR_FIXED` column.
        - `contains_yellow` | Variable indicating whether 'tan/yellow/golden'
                                appear in the `COLOR_FIXED` column.
    '''
    cleaned_df = df.copy(deep = True)

    # Uncomment the below line if you want to make the dataframes easier to
    # work with, for testing purposes
    #    cleaned_df = cleaned_df[['DOG NAME', 'BREED MIXES', 'COLOR']]

    # Start by making a copy of our 'COLOR' column
    cleaned_df['COLOR_FIXED'] = cleaned_df['COLOR']

    # Before we start, there are some columns that need to be reassigned)
    # because they actually belong in another column but somehow got read into
    # the color column. Pull the correct value from the 'BREED MIXES' column.
    # Note: Some are 'nan' so replace these with False
    cleaned_df.loc[
        cleaned_df['COLOR'].str.lower().str.contains('screen', na = False), 'COLOR_FIXED'] = \
    cleaned_df['BREED MIXES']
    cleaned_df.loc[
        cleaned_df['COLOR'].str.lower().str.contains('walks alone', na = False), 'COLOR_FIXED'] = \
    cleaned_df['BREED MIXES']
    cleaned_df.loc[cleaned_df['COLOR'].str.lower().str.contains('month old puppy').replace(np.nan,
                                                                                                       False), 'COLOR_FIXED'] = \
    cleaned_df['BREED MIXES']
    cleaned_df.loc[cleaned_df['COLOR'].str.lower().str.contains('new this week').replace(np.nan,
                                                                                                     False), 'COLOR_FIXED'] = \
    cleaned_df['BREED MIXES']
    cleaned_df.loc[
        cleaned_df['COLOR'].str.lower().str.contains('kids', na = False), 'COLOR_FIXED'] = \
    cleaned_df['BREED MIXES']
    cleaned_df.loc[
        cleaned_df['COLOR'].str.lower().str.contains('energy', na = False), 'COLOR_FIXED'] = \
    cleaned_df['BREED MIXES']
    cleaned_df.loc[cleaned_df['COLOR'].str.lower().str.contains('HW', na = False), 'COLOR_FIXED'] = \
    cleaned_df['BREED MIXES']
    cleaned_df.loc[
        cleaned_df['COLOR'].str.lower().str.contains('nervous', na = False), 'COLOR_FIXED'] = \
    cleaned_df['BREED MIXES']
    cleaned_df.loc[
        cleaned_df['COLOR'].str.lower().str.contains('retriever', na = False), 'COLOR_FIXED'] = \
    cleaned_df['BREED MIXES']

    # Start by replacing any combination characters (/, w/, with, &, and) with ','
    cleaned_df['COLOR_FIXED'] = cleaned_df['COLOR_FIXED'].str.lower().replace('&', ',', regex=True)
    cleaned_df['COLOR_FIXED'] = cleaned_df['COLOR_FIXED'].str.replace(', and', ',', regex=True)
    cleaned_df['COLOR_FIXED'] = cleaned_df['COLOR_FIXED'].str.replace('and', ',', regex=True)
    cleaned_df['COLOR_FIXED'] = cleaned_df['COLOR_FIXED'].str.replace('with', ',', regex=True)
    cleaned_df['COLOR_FIXED'] = cleaned_df['COLOR_FIXED'].str.replace(' w/', ', ', regex=True)
    cleaned_df['COLOR_FIXED'] = cleaned_df['COLOR_FIXED'].str.replace(' w /', ', ', regex=True)
    cleaned_df['COLOR_FIXED'] = cleaned_df['COLOR_FIXED'].str.replace(' w ', ', ', regex=True)
    cleaned_df['COLOR_FIXED'] = cleaned_df['COLOR_FIXED'].str.replace('/', ', ', regex=True)

    # If there's a 'or', then remove whatever comes after this so we
    # don't get tripped up. Make sure to include a space after 'or' so
    # we don't accidentally get any Orange-colored dogs messed up
    cleaned_df['COLOR_FIXED'] = cleaned_df['COLOR_FIXED'].str.replace('or ', ', ', regex=True)

    # Replace hyphens '-' with np.nan
    cleaned_df.loc[cleaned_df['COLOR_FIXED'] == '-', 'COLOR_FIXED'] = np.nan

    # A lot of dogs are 'brown (chocolate)'. Replace this with just 'brown'
    cleaned_df['COLOR_FIXED'] = cleaned_df['COLOR_FIXED'].str.replace('brown \(chocolate\)', 'brown', regex=True)

    # Format the commas a bit better
    cleaned_df['COLOR_FIXED'] = cleaned_df['COLOR_FIXED'].str.replace(' , ', ',', regex=True)
    cleaned_df['COLOR_FIXED'] = cleaned_df['COLOR_FIXED'].str.replace(', ,', ', ', regex=True)
    cleaned_df['COLOR_FIXED'] = cleaned_df['COLOR_FIXED'].str.replace(',  ', ',', regex=True)
    cleaned_df['COLOR_FIXED'] = cleaned_df['COLOR_FIXED'].str.replace(', ', ',', regex=True)
    cleaned_df['COLOR_FIXED'] = cleaned_df['COLOR_FIXED'].str.replace(',', ', ', regex=True)

    # Remove extra whitespace from the color column
    cleaned_df['COLOR_FIXED'] = cleaned_df['COLOR_FIXED'].str.strip()

    # Create a new column to figure out whether a dog is multi-colored or not
    # We'll do so by creating booleans to detect whether or not a dog is 'tricolor'
    # or has a comma (from earlier) in it, denoting multiple colors.
    # Note: Some dogs have 'nan' (no) color, so replace these with falses
    tri_color_bool = (cleaned_df['COLOR_FIXED'].str.contains('tri', na = False) & \
                      cleaned_df['COLOR_FIXED'].str.contains('color', na = False))
    multi_color_bool = (cleaned_df['COLOR_FIXED'].str.contains(',', na = False))

    # Create our multi-color column accordingly
    cleaned_df.loc[tri_color_bool | multi_color_bool, 'multi_color'] = 1
    cleaned_df.loc[~(tri_color_bool) & ~(multi_color_bool), 'multi_color'] = 0
    # For anyone who didn't have a color to begin with, make MULTI_COLOR = 'nan'
    cleaned_df.loc[cleaned_df['COLOR_FIXED'].isna(), 'multi_color'] = np.nan

    # For interest purposes, let's also create a column for the number of colors
    # in a dog's mane. If it's a tri-color dog, we'll just assume 3.
    cleaned_df.loc[:, 'num_colors'] = cleaned_df['COLOR_FIXED'].str.count(',') + 1
    cleaned_df.loc[tri_color_bool, 'num_colors'] = 3

    # Create indicator columns for whether or not a dog has the following colors:
    #   1. Black
    #   2. White
    #   3. Tan/Yellow/Golden
    # This is to try to prove the hypothesis that black dogs are adopted less/
    # returned more frequently than other-colored dogs.
    cleaned_df.loc[
        (cleaned_df['COLOR_FIXED'].str.contains('black')) & ~(cleaned_df['COLOR_FIXED'].isna()), 'contains_black'] = 1
    cleaned_df.loc[~(cleaned_df['COLOR_FIXED'].str.contains('black', na = False)) & ~(
        cleaned_df['COLOR_FIXED'].isna()), 'contains_black'] = 0
    cleaned_df.loc[
        (cleaned_df['COLOR_FIXED'].str.contains('white')) & ~(cleaned_df['COLOR_FIXED'].isna()), 'contains_white'] = 1
    cleaned_df.loc[~(cleaned_df['COLOR_FIXED'].str.contains('white', na = False)) & ~(
        cleaned_df['COLOR_FIXED'].isna()), 'contains_white'] = 0
    # Create a boolean for tan/yellow/golden
    yellow_bool = (cleaned_df['COLOR_FIXED'].str.contains('tan')) | (
        cleaned_df['COLOR_FIXED'].str.contains('yellow')) | (cleaned_df['COLOR_FIXED'].str.contains('golden'))
    cleaned_df.loc[(yellow_bool) & ~(cleaned_df['COLOR_FIXED'].isna()), 'contains_yellow'] = 1
    cleaned_df.loc[~(yellow_bool) & ~(cleaned_df['COLOR_FIXED'].isna()), 'contains_yellow'] = 0

    return cleaned_df


def clean_sex(df):

    # update adoption list gender column to Male or Female and remove typos/misspellings
    cleaned_df = df.copy(deep = True)
    df_obj = cleaned_df.select_dtypes(['object'])
    cleaned_df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    cleaned_df.loc[cleaned_df["SEX"].str.contains("f", case = False, na = False, regex=False), 'SEX'] = "Female"
    cleaned_df.loc[cleaned_df["SEX"].str.startswith("m", na = False), 'SEX'] = "Male"
    cleaned_df.loc[cleaned_df["SEX"].str.startswith("M", na = False), 'SEX'] = "Male"
    one_hot = pd.get_dummies(cleaned_df.SEX, prefix='SEX')
    cleaned_df = cleaned_df.join(one_hot)

    return cleaned_df


def clean_weight(df):
    # update adoption list weight column
    cleaned_df = df.copy(deep = True)
    cleaned_df["WEIGHT2"] = cleaned_df.loc[~cleaned_df["WEIGHT"].str.contains("/", case=False, na=False, regex=False), 'WEIGHT']
    cleaned_df["WEIGHT2"] = cleaned_df["WEIGHT2"].str.extract("([-+]?\d*\.\d+|[-+]?\d+)").astype(float)
    cleaned_df.drop(columns = ["WEIGHT"])
    cleaned_df.rename({"WEIGHT2": "WEIGHT"})
    cleaned_df.drop(columns=["WEIGHT2"])

    return cleaned_df


def clean_age(df):
    # update adoption list age column, provide age in days
    cleaned_df = df.copy(deep = True)

    cleaned_df['AGE'] = cleaned_df['AGE'].str.replace(',', '.', regex=True)
    cleaned_df['AGE'] = cleaned_df['AGE'].str.split('as of').str[0]
    cleaned_df['AGE'] = cleaned_df['AGE'].str.split('a/o').str[0]
    cleaned_df['AGE'] = cleaned_df['AGE'].str.split('A/O').str[0]
    cleaned_df['AGE'] = cleaned_df['AGE'].str.split('As of').str[0]
    cleaned_df['AGE'] = cleaned_df['AGE'].str.split('As Of').str[0]
    cleaned_df['AGE'] = cleaned_df['AGE'].str.split('A/0').str[0]

    cleaned_df['DOB2'] = cleaned_df['AGE'].str.extract(r"(\d{1,2}[\/. ](?:\d{1,2}|January|Jan)[\/. ]\d{2}(?:\d{2})?)")
    cleaned_df['DOB2'] = cleaned_df['AGE'].str.extract(r"(\d{1,4}[-. ](?:\d{1,2}|January|Jan)[-. ]\d{2}(?:\d{2})?)")
    cleaned_df['Date of Birth'] = cleaned_df['Date of Birth'].fillna(cleaned_df.pop('DOB2'))

    cleaned_df['Date of Birth'] = pd.to_datetime(cleaned_df['Date of Birth'], errors='coerce')
    cleaned_df['Adopted On'] = pd.to_datetime(cleaned_df['Adopted On'], errors='coerce')
    cleaned_df["Age at Adoption (days)"] = (cleaned_df['Adopted On'] - cleaned_df['Date of Birth']).dt.days

    #cleaned_df.loc[cleaned_df["Age at Adoption (days)"].isnull(), "Age at Adoption (days)"] = (pd.to_datetime("today") - cleaned_df['Date of Birth']).dt.days

    cleaned_df["AGE_TYPE"] = cleaned_df["AGE"].replace('(\d)', '', regex=True)
    cleaned_df.loc[cleaned_df.puppy_screen == 1, 'AGE_TYPE'] = 52
    cleaned_df['AGE_TYPE'] = cleaned_df['AGE_TYPE'].fillna(cleaned_df.pop('AGE MEASURE'))
    cleaned_df.loc[cleaned_df["AGE_TYPE"].str.contains("mo", case=False, na=False, regex=False), 'AGE_TYPE'] = 30
    cleaned_df.loc[cleaned_df["AGE_TYPE"].str.contains("ye", case=False, na=False, regex=False), 'AGE_TYPE'] = 365
    cleaned_df.loc[cleaned_df["AGE_TYPE"].str.contains("yr", case=False, na=False, regex=False), 'AGE_TYPE'] = 365
    cleaned_df.loc[cleaned_df["AGE_TYPE"].str.contains("we", case=False, na=False, regex=False), 'AGE_TYPE'] = 52
    cleaned_df.loc[cleaned_df["AGE_TYPE"].str.contains("day", case=False, na=False, regex=False), 'AGE_TYPE'] = 1
    cleaned_df.loc[~cleaned_df.AGE_TYPE.isin([30,1,52,365]), "AGE_TYPE"] = 365

    cleaned_df['AGE_TYPE'] = cleaned_df['AGE_TYPE'].fillna(365)

    cleaned_df['AGE2'] = cleaned_df['AGE'].str.extract('([0-9][,.]*[0-9]*)').astype(float)
    cleaned_df.loc[cleaned_df.AGE2 >= 12, 'AGE_TYPE'] = 52
    cleaned_df["AGE_FIXED"] = cleaned_df["AGE2"] * cleaned_df["AGE_TYPE"].astype(float)

    cleaned_df.loc[cleaned_df["Age at Adoption (days)"] < 0] = np.nan
    cleaned_df["Age at Adoption (days)"] = cleaned_df["Age at Adoption (days)"].fillna(cleaned_df.pop('AGE_FIXED'))

    cleaned_df.loc[cleaned_df.ID == "OWNR-FD-19-1408", "Age at Adoption (days)"] = 728
    cleaned_df.loc[cleaned_df["Adopted On"].isna(), "Age at Adoption (days)"] = np.nan

    return cleaned_df


def clean_BSW(df):
    # update the BSW Column to be a boolean yes/no for screen and warning
    cleaned_df = df.copy(deep=True)
    cleaned_df.loc[(cleaned_df['BS/W'].str.contains('screen', case=False, na=False)) | \
                   (cleaned_df['BS/W'].str.contains('BS', case=False, na=False)) | \
                   (cleaned_df['BS/W'].str.contains('screeen', case=False, na=False)) & \
                   ~(cleaned_df['BS/W'].str.contains('GS', case=False, na=False)),
                   'BULLY_SCREEN'] = 1
    cleaned_df.loc[(cleaned_df['BEHAVIORAL NOTES'].str.contains('bully', case=False, na=False)) & \
                   (cleaned_df['BEHAVIORAL NOTES'].str.contains('screen', case= False, na=False)) & \
                   ~(cleaned_df['BEHAVIORAL NOTES'].str.contains('puppy screen', case= False, na=False)),
                   'BULLY_SCREEN'] = 1
    cleaned_df.loc[(cleaned_df['BS/W'].str.contains('warning', case=False, na=False)) | \
                   (cleaned_df['BS/W'].str.contains('BW', case=False, na=False)),
                   'BULLY_WARNING'] = 1
    cleaned_df.loc[(cleaned_df['BEHAVIORAL NOTES'].str.contains('bully', case=False, na=False)) & \
                   (cleaned_df['BEHAVIORAL NOTES'].str.contains('warning', case=False, na=False)),
                   'BULLY_WARNING'] = 1
    cleaned_df.loc[~(cleaned_df['BEHAVIORAL NOTES'].str.contains('bully', case=False, na=False)) & \
                   (cleaned_df['BEHAVIORAL NOTES'].str.contains('warning', case=False, na=False)) | \
                   (cleaned_df['BEHAVIORAL NOTES'].str.contains('restriction', case=False, na=False)),
                   'OTHER_WARNING'] = 1
    cleaned_df.loc[(cleaned_df['BS/W'].str.contains('GSD', case=False, na=False)) | \
                   (cleaned_df['BS/W'].str.contains('GS', case=False, na=False)) | \
                   (cleaned_df['BS/W'].str.contains('German', case=False, na=False)) | \
                   (cleaned_df['BS/W'].str.contains('husky', case=False, na=False)) ,
                   'OTHER_WARNING'] = 1
    cleaned_df['BULLY_SCREEN'] = cleaned_df['BULLY_SCREEN'].fillna(0)
    cleaned_df['BULLY_WARNING'] = cleaned_df['BULLY_WARNING'].fillna(0)
    cleaned_df['OTHER_WARNING'] = cleaned_df['OTHER_WARNING'].fillna(0)

    return cleaned_df

def clean_cats(df):
    # update the BSW Column to be a boolean yes/no for lived with cats and passed shelter test
    cleaned_df = df.copy(deep=True)

    cleaned_df['CATS'] = cleaned_df['CATS'].str.strip()

    cleaned_df["CATS"].str.replace(r'\bn\b', "No", regex=True)
    cleaned_df["CATS"].str.replace(r'\bN\b', "No", regex=True)

    cleaned_df.loc[(cleaned_df['CATS'].str.contains('y', case=False, na=False)) | \
                   (cleaned_df['CATS'].str.contains('yes', case=False, na=False)) | \
                   (cleaned_df['CATS'].str.contains('live', case=False, na=False)),
                   'CATS_LIVED_WITH'] = 1
    cleaned_df.loc[(cleaned_df['CATS'].str.contains(r'\bNo\b', case=False, na=False)) | \
                    (cleaned_df['CATS'].str.contains(r'\bn\b', case=False, na=False)),
                   'CATS_LIVED_WITH'] = 0
    cleaned_df.loc[(cleaned_df['CATS'].str.contains('passed', case=False, na=False)) | \
                   (cleaned_df['CATS'].str.contains('test', case=False, na=False)) | \
                   (cleaned_df['CATS'].str.contains('good', case=False, na=False)) | \
                   (cleaned_df['CATS'].str.contains('curious', case=False, na=False)) | \
                   (cleaned_df['CATS'].str.contains('liv', case=False, na=False)) | \
                   (cleaned_df['CATS'].str.contains('y', case=False, na=False)) | \
                   (cleaned_df['CATS'].str.contains('yes', case=False, na=False)) ,
                   'CATS_TEST'] = 1
    cleaned_df.loc[(cleaned_df['CATS'].str.contains('caution', case=False, na=False)) | \
                   (cleaned_df['CATS'].str.contains(r'\bNo\b', case=False, na=False)) | \
                   (cleaned_df['CATS'].str.contains('debatable', case=False, na=False)),
                   'CATS_TEST'] = 0
    cleaned_df.loc[(cleaned_df['BEHAVIORAL NOTES'].str.contains('good with cats', case=False, na=False)) | \
                   (cleaned_df['BEHAVIORAL NOTES'].str.contains('good with cats and kids', case=False, na=False)) | \
                   (cleaned_df['BEHAVIORAL NOTES'].str.contains('good with cats & kids', case=False, na=False)),
                   'CATS_TEST'] = 1
    cleaned_df.loc[(cleaned_df['BEHAVIORAL NOTES'].str.contains('good with cats', case=False, na=False)) | \
                   (cleaned_df['BEHAVIORAL NOTES'].str.contains('good with cats and kids', case=False, na=False)) | \
                   (cleaned_df['BEHAVIORAL NOTES'].str.contains('good with cats & kids', case=False, na=False)),
                   'CATS_LIVED_WITH'] = 1
    cleaned_df.loc[(cleaned_df['BEHAVIORAL NOTES'].str.contains('no cats', case=False, na=False)),
                   'CATS_LIVED_WITH'] = 0
    cleaned_df.loc[(cleaned_df['BEHAVIORAL NOTES'].str.contains('no cats', case=False, na=False)),
                   'CATS_TEST'] = 0

    # cleaned_df['CATS_LIVED_WITH'] = cleaned_df['CATS_LIVED_WITH'].fillna(0)
    # cleaned_df['CATS_TEST'] = cleaned_df['CATS_TEST'].fillna(0)

    return cleaned_df


def clean_kids(df):
    # update the kids Column to be a boolean yes/no for kids or not
    cleaned_df = df.copy(deep=True)

    cleaned_df.loc[(cleaned_df['KIDS'].str.contains('y', case=False, na=False)) | \
                   (cleaned_df['KIDS'].str.contains('yes', case=False, na=False)) | \
                   (cleaned_df['KIDS'].str.contains('liv', case=False, na=False)),
                   'KIDS_FIXED'] = 1
    cleaned_df.loc[(cleaned_df['KIDS'].str.contains('no', case=False, na=False)) | \
                   (cleaned_df['KIDS'].str.contains('n', case=False, na=False)) | \
                   (cleaned_df['KIDS'].str.contains('caution', case=False, na=False)),
                   'KIDS_FIXED'] = 0
    cleaned_df.loc[(cleaned_df['BEHAVIORAL NOTES'].str.contains('no kids', case=False, na=False)),
                   'KIDS_FIXED'] = 0
    cleaned_df.loc[(cleaned_df['BEHAVIORAL NOTES'].str.contains('good with kids', case=False, na=False)) | \
                    (cleaned_df['BEHAVIORAL NOTES'].str.contains('good with cats and kids', case=False, na=False))| \
                    (cleaned_df['BEHAVIORAL NOTES'].str.contains('good with cats & kids', case=False, na=False)),
                   'KIDS_FIXED'] = 1

    return cleaned_df


def clean_dogs(df):
    # update the dogs Column to be a boolean yes/no for dogs, as well as recommended/required
    cleaned_df = df.copy(deep=True)

    cleaned_df.loc[(cleaned_df['DOGS IN HOME'].str.contains('y', case=False, na=False)) | \
                   (cleaned_df['DOGS IN HOME'].str.contains('yes', case=False, na=False)) | \
                   (cleaned_df['DOGS IN HOME'].str.contains('liv', case=False, na=False)) | \
                   (cleaned_df['DOGS IN HOME'].str.contains('required', case=False, na=False)) | \
                   (cleaned_df['DOGS IN HOME'].str.contains('recommended', case=False, na=False)) | \
                   (cleaned_df['DOGS IN HOME'].str.contains('need', case=False, na=False)),
                   'DOGS_IN_HOME'] = 1
    cleaned_df.loc[(cleaned_df['DOGS IN HOME'].str.contains('dog', case=False, na=False)) & \
                   (cleaned_df['DOGS IN HOME'].str.contains('only', case=False, na=False)),
                   'DOGS_IN_HOME'] = 0
    cleaned_df.loc[(cleaned_df['DOGS IN HOME'].str.contains('required', case=False, na=False)) | \
                   (cleaned_df['DOGS IN HOME'].str.contains('recommended', case=False, na=False)) | \
                   (cleaned_df['DOGS IN HOME'].str.contains('need', case=False, na=False)),
                   'DOGS_REQ'] = 1
    cleaned_df.loc[(cleaned_df['BEHAVIORAL NOTES'].str.contains('dog', case=False, na=False)) & \
                   (cleaned_df['BEHAVIORAL NOTES'].str.contains('req', case=False, na=False)) | \
                   (cleaned_df['BEHAVIORAL NOTES'].str.contains('rec', case=False, na=False)),
                   'DOGS_REQ'] = 1
    cleaned_df.loc[(cleaned_df['BEHAVIORAL NOTES'].str.contains('dog', case=False, na=False)) & \
                   (cleaned_df['BEHAVIORAL NOTES'].str.contains('only', case=False, na=False)),
                   'DOGS_IN_HOME'] = 0

    cleaned_df['DOGS_REQ'] = cleaned_df['DOGS_REQ'].fillna(0)

    return cleaned_df

def clean_HW_FT(df):
    # update the HW and FT Columns to be a boolean yes/no
    cleaned_df = df.copy(deep=True)
    cleaned_df[['HW_FIXED', 'FT_FIXED']] = cleaned_df[['HW', 'FT']].notnull().astype(int)
    cleaned_df.loc[cleaned_df["HW"] == "No", "HW_FIXED"] = 0
    cleaned_df.loc[cleaned_df["FT"] == "No", "FT_FIXED"] = 0

    return cleaned_df

adopts_clean = clean_color(adopts)
adopts_clean = clean_weight(adopts_clean)
adopts_clean = clean_sex(adopts_clean)
adopts_clean = clean_BSW(adopts_clean)
adopts_clean = clean_cats(adopts_clean)
adopts_clean = clean_kids(adopts_clean)
adopts_clean = clean_dogs(adopts_clean)
adopts_clean = clean_HW_FT(adopts_clean)

print(adopts_clean.head())


#%% Clean up the `BREED MIXES` column
def clean_breed_mixes(df):
    '''
    This function is meant to clean up the `BREED MIXES` column, which denotes 
    a dog's breed. Because there are so many combinations of different breeds for
    dogs, the main purpose is to create multiple indicator columns for whether or
    not a dog is a certain breed (Lab/Retriever, Shepherd, or Other).
    
    '''
    cleaned_df = df.copy(deep = True)

    # Uncomment the below line if you want to make the dataframes easier to
    # work with, for testing purposes
    # cleaned_df = cleaned_df[['DOG NAME', 'BREED MIXES', 'SECONDARY BREED', 'MIX']]

    # Create a copy of the column that's proper title case
    cleaned_df['BREED_MIXES_FIXED'] = cleaned_df['BREED MIXES'].str.title()
    
    # Start by replacing any combination characters (/, w/, with, &, and) with ','
    cleaned_df['BREED_MIXES_FIXED'] = cleaned_df['BREED_MIXES_FIXED'].str.replace('&', ',', regex=True)
    cleaned_df['BREED_MIXES_FIXED'] = cleaned_df['BREED_MIXES_FIXED'].str.replace(', and', ',', regex=True)
    cleaned_df['BREED_MIXES_FIXED'] = cleaned_df['BREED_MIXES_FIXED'].str.replace('and ', ',', regex=True)
    cleaned_df['BREED_MIXES_FIXED'] = cleaned_df['BREED_MIXES_FIXED'].str.replace('with', ',', regex=True)
    cleaned_df['BREED_MIXES_FIXED'] = cleaned_df['BREED_MIXES_FIXED'].str.replace(' w/', ', ', regex=True)
    cleaned_df['BREED_MIXES_FIXED'] = cleaned_df['BREED_MIXES_FIXED'].str.replace(' w /', ', ', regex=True)
    cleaned_df['BREED_MIXES_FIXED'] = cleaned_df['BREED_MIXES_FIXED'].str.replace(' w ', ', ', regex=True)
    cleaned_df['BREED_MIXES_FIXED'] = cleaned_df['BREED_MIXES_FIXED'].str.replace('/', ', ', regex=True)
    
    # Format the commas a bit better
    cleaned_df['BREED_MIXES_FIXED'] = cleaned_df['BREED_MIXES_FIXED'].str.replace(' , ', ',', regex=True)
    cleaned_df['BREED_MIXES_FIXED'] = cleaned_df['BREED_MIXES_FIXED'].str.replace(', ,', ', ', regex=True)
    cleaned_df['BREED_MIXES_FIXED'] = cleaned_df['BREED_MIXES_FIXED'].str.replace(',  ', ',', regex=True)
    cleaned_df['BREED_MIXES_FIXED'] = cleaned_df['BREED_MIXES_FIXED'].str.replace(', ', ',', regex=True)
    cleaned_df['BREED_MIXES_FIXED'] = cleaned_df['BREED_MIXES_FIXED'].str.replace(',', ', ', regex=True)

    # Remove extra whitespace from the color column
    cleaned_df['BREED_MIXES_FIXED'] = cleaned_df['BREED_MIXES_FIXED'].str.strip()

    # Create indicator columns for whether or not a dog is one of the following breeds:
    #   1. Retriever/Lab
    #   2. Shepherd
    #   3. Other Breed (Not Retriever/Lab/Shepherd)
    cleaned_df.loc[(cleaned_df['BREED_MIXES_FIXED'].str.contains('Retriever', na = False)) | \
                   (cleaned_df['BREED_MIXES_FIXED'].str.contains('Lab', na = False)), 
                   'is_retriever'] = 1
    cleaned_df.loc[~(cleaned_df['BREED_MIXES_FIXED'].str.contains('Retriever', na = False)) & \
                   ~(cleaned_df['BREED_MIXES_FIXED'].str.contains('Lab', na = False)), 
                   'is_retriever'] = 0

    cleaned_df.loc[(cleaned_df['BREED_MIXES_FIXED'].str.contains('Shep', na = False)), 
                   'is_shepherd'] = 1
    cleaned_df.loc[~(cleaned_df['BREED_MIXES_FIXED'].str.contains('Shep', na = False)), 
                   'is_shepherd'] = 0
    # Build the `is_other_breed` column based on the previous two booleans
    cleaned_df.loc[(cleaned_df['is_retriever'] == 0) & \
                   (cleaned_df['is_shepherd'] == 0), 'is_other_breed'] = 1
    cleaned_df.loc[(cleaned_df['is_retriever'] == 1) | \
                   (cleaned_df['is_shepherd'] == 1) | \
                   (cleaned_df['BREED_MIXES_FIXED'].isna()), 'is_other_breed'] = 0

    
    return cleaned_df


adopts_clean2 = clean_breed_mixes(adopts_clean)

print(adopts_clean2['BREED MIXES'].value_counts())
print(adopts_clean2['BREED_MIXES_FIXED'].value_counts())
print(adopts_clean2['is_retriever'].value_counts())
print(adopts_clean2['is_shepherd'].value_counts())
print(adopts_clean2['is_other_breed'].value_counts())



#%% Clean up the `MIX` column
def clean_mix(df):
    '''
    This function is meant to clean up the MIX column, which denotes whether a
    dog is mixed breed. 
    
    Some of the issues with this column include:
        - The word 'Mix' is not standard title-case (i.e. 'mix' instead of 'Mix')
        - Some dogs should be denoted as a 'Mix' because they're multiple breeds
            but the `MIX` column is null
            
    Lastly, we create a boolean (1 for 'Mix', 0 otherwise).
    
    '''
    cleaned_df = df.copy(deep = True)

    # Uncomment the below line if you want to make the dataframes easier to
    # work with, for testing purposes
    # cleaned_df = cleaned_df[['DOG NAME', 'BREED MIXES', 'SECONDARY BREED', 'MIX']]

    # Start by making a copy of our 'COLOR' column and make it
    # title case so it's standardized
    cleaned_df['MIX_FIXED'] = cleaned_df['MIX'].str.title()
    
    # Any values in the `BREED MIXES` column that contain a comma
    # are likely a mix as well, so assign those the value 'Mix'
    cleaned_df.loc[cleaned_df['BREED_MIXES_FIXED'].str.contains(',', na = False), 'MIX_FIXED'] = 'Mix'

    # Some dog breeds actually have the word 'mix' in them (i.e. 'Lab Mix').
    # Let's assign these as 'Mix' as well
    cleaned_df.loc[cleaned_df['BREED_MIXES_FIXED'].str.lower().str.contains('mix', case = False, na = False), 'MIX_FIXED'] = 'Mix'
    
    # One-hot encode the new `MIX_FIXED` column
    cleaned_df['MIX_BOOL'] = cleaned_df['MIX_FIXED'].apply(lambda x: 1 if x == 'Mix' else 0)
    # Make sure NA's are represented by NA and not 0
    cleaned_df.loc[(cleaned_df['BREED_MIXES_FIXED'].isna()) & (cleaned_df['MIX_FIXED'].isna()), 'MIX_BOOL'] = np.nan
    
    return cleaned_df

adopts_clean3 = clean_mix(adopts_clean2)

print(adopts_clean3['MIX'].value_counts())
print(adopts_clean3['MIX_FIXED'].value_counts())
print(adopts_clean3['MIX_BOOL'].value_counts())

#%% Clean BEHAVIORAL NOTES column
def clean_behav_notes(df):
    '''
    This function is meant to clean up the `BEHAVIORAL NOTES` column, which
    describes any comments about a dog's temperament. There's not *too* much
    cleaning here -- just some general standardization --, but there's a lot of
    good notes about dogs' temperaments denoted here. To pull out this data, we'll:
        
        1. Split the behavioral notes on semicolon ';' and then create a counter
            for how many behavioral issues/notes a dog has.
        2. Create indicators for each major behavior.
    
    '''
    cleaned_df = df.copy(deep = True)

    cleaned_df['BEHAVIORAL_NOTES_FIXED'] = cleaned_df['BEHAVIORAL NOTES'].str.title().str.strip()
    
    # Uncomment the below line if you want to make the dataframes easier to
    # work with, for testing purposes
    # cleaned_df = cleaned_df[['DOG NAME', 'BEHAVIORAL NOTES', 'BEHAVIORAL_NOTES_FIXED']]
    
    # To see unique values in this column, reference the below lines, which
    # split the behavioral notes by semicolon ';', pivot the notes into a single
    # dataframe column, and then finds unique values.
    behavioral_notes = cleaned_df[['BEHAVIORAL_NOTES_FIXED']].applymap(lambda x: x.split('; ') if isinstance(x, str) else [x])
    behavioral_notes_pivoted = pd.DataFrame([j for i in behavioral_notes.values for j in product(*i)], columns = behavioral_notes.columns)
    print(behavioral_notes_pivoted['BEHAVIORAL_NOTES_FIXED'].value_counts())
    behavioral_notes_counts = behavioral_notes_pivoted['BEHAVIORAL_NOTES_FIXED'].value_counts()
    behavioral_notes_pivoted['BEHAVIORAL_NOTES_FIXED'] = behavioral_notes_pivoted['BEHAVIORAL_NOTES_FIXED'].str.strip()
    behavioral_notes_pivoted = behavioral_notes_pivoted.drop_duplicates()
    
    # Let's count up the number of behavioral notes for a given dog using the 
    # splitting we just did
    cleaned_df['num_behav_issues'] = behavioral_notes['BEHAVIORAL_NOTES_FIXED'].apply(lambda x: len(x))
    
    # Create indicator columns for the following variables
    #   1. Puppy Screen
    #   2. New This Week
    #   3. Walks Alone Not Enough (Needs Running or Playtime)
    #   4. Caution on Apartments
    #   5. High/Medium Energy/Energetic
    #   6. No Kids Under 12 Due to Shyness / Commitment to Socialization
    #   7. Commitment to Training / Needs Training
    cleaned_df.loc[cleaned_df['BEHAVIORAL_NOTES_FIXED'].str.contains('Puppy Screen', na = False), 
                   'puppy_screen'] = 1
    cleaned_df.loc[cleaned_df['puppy_screen'] != 1, 
                   'puppy_screen'] = 0
    cleaned_df.loc[cleaned_df['BEHAVIORAL_NOTES_FIXED'].str.contains('New This Week', na = False), 
                   'new_this_week'] = 1
    cleaned_df.loc[cleaned_df['new_this_week'] != 1, 
                   'new_this_week'] = 0
    cleaned_df.loc[cleaned_df['BEHAVIORAL_NOTES_FIXED'].str.contains('Walks Alone Not Enough', na = False), 
                   'needs_play'] = 1
    cleaned_df.loc[cleaned_df['needs_play'] != 1, 
                   'needs_play'] = 0
    cleaned_df.loc[cleaned_df['BEHAVIORAL_NOTES_FIXED'].str.contains('Caution On Apartments', na = False), 
                   'no_apartments'] = 1
    cleaned_df.loc[cleaned_df['no_apartments'] != 1, 
                   'no_apartments'] = 0
    cleaned_df.loc[cleaned_df['BEHAVIORAL_NOTES_FIXED'].str.contains('Energy', na = False), 
                   'energetic'] = 1
    cleaned_df.loc[cleaned_df['energetic'] != 1, 
                   'energetic'] = 0
    cleaned_df.loc[cleaned_df['BEHAVIORAL_NOTES_FIXED'].str.contains('Shy', na = False) | \
                   cleaned_df['BEHAVIORAL_NOTES_FIXED'].str.contains('Socialization'), 
                   'shyness'] = 1
    cleaned_df.loc[cleaned_df['shyness'] != 1, 
                   'shyness'] = 0
    cleaned_df.loc[cleaned_df['BEHAVIORAL_NOTES_FIXED'].str.contains('Training', na = False), 
                   'needs_training'] = 1
    cleaned_df.loc[cleaned_df['needs_training'] != 1, 
                   'needs_training'] = 0
    
    return cleaned_df, behavioral_notes_pivoted, behavioral_notes_counts

adopts_clean4, behavioral_notes_pivoted, behavioral_notes_counts = clean_behav_notes(adopts_clean3)

# print(adopts_clean4['BEHAVIORAL NOTES'].value_counts())
# print(adopts_clean4['BEHAVIORAL_NOTES_FIXED'].value_counts())
print('\nnum_behav_issues\n', adopts_clean4['num_behav_issues'].value_counts())
print('\npuppy_screen\n', adopts_clean4['puppy_screen'].value_counts())
print('\nnew_this_week\n', adopts_clean4['new_this_week'].value_counts())
print('\nneeds_play\n', adopts_clean4['needs_play'].value_counts())
print('\nno_apartments\n', adopts_clean4['no_apartments'].value_counts())
print('\nenergetic\n', adopts_clean4['energetic'].value_counts())
print('\nshyness\n', adopts_clean4['shyness'].value_counts())
print('\nneeds_training\n', adopts_clean4['needs_training'].value_counts())


#%% Clean MEDICAL NOTES column
def clean_med_notes(df):
    '''
    This function is meant to clean up the `MEDICAL NOTES` column, which
    includes comments about a dog's health. 
    
    '''
    cleaned_df = df.copy(deep = True)
    
    # Uncomment the below line if you want to make the dataframes easier to
    # work with, for testing purposes
    # cleaned_df = cleaned_df[['DOG NAME', 'MEDICAL NOTES']]
    
    # Create indicator columns for the following medical conditions
    #   0. ANY medical condition (i.e. any value in the MEDICAL NOTES field)
    #   1. Diarrhea
    #   2. Ehrlichia (the disease Ehrlichiosis)/Lyme transmitted by ticks
    #   3. Upper Respiratory Infection (URI)
    #   4. Ear Infection
    #   5. Tapeworm
    #   6. General Infection
    #   7. Demodex (Demodectic mange aka mites)
    #   8. Car Sick
    #   9. No exposure to dog parks yet (indicating a lack of exposure/immunity to potential
    #       diseases)
    #   10. Leg issues (amputated, sprained, swollen, lesions, etc.) :'(
    #   11. Anaplasmosis
    #   12. Teeth/dental issues
    #   13. Weight issues (overweight)
    #   14. Hair loss
    #   15. Already received treatments/vaccinations
    cleaned_df.loc[~(cleaned_df['MEDICAL NOTES'].isna()), 
                   'has_med_issues'] = 1
    cleaned_df.loc[cleaned_df['has_med_issues'] != 1, 
                   'has_med_issues'] = 0
    cleaned_df.loc[cleaned_df['MEDICAL NOTES'].str.contains('diarrhea', case = False, na = False), 
                   'diarrhea'] = 1
    cleaned_df.loc[cleaned_df['diarrhea'] != 1, 
                   'diarrhea'] = 0
    cleaned_df.loc[cleaned_df['MEDICAL NOTES'].str.contains('ehrlichi', case = False, na = False) |
                   cleaned_df['MEDICAL NOTES'].str.contains('lyme', case = False, na = False) |
                   cleaned_df['MEDICAL NOTES'].str.contains('ticks', case = False, na = False), 
                   'ehrlichia'] = 1
    cleaned_df.loc[cleaned_df['ehrlichia'] != 1, 
                   'ehrlichia'] = 0
    # URI: watch out for 'during'
    cleaned_df.loc[cleaned_df['MEDICAL NOTES'].str.contains('URI', case = True, na = False) |
                   cleaned_df['MEDICAL NOTES'].str.contains('respiratory', case = False, na = False), 
                   'uri'] = 1
    cleaned_df.loc[cleaned_df['uri'] != 1, 
                   'uri'] = 0
    # Ear Infection: watch out for 'clear'
    cleaned_df.loc[cleaned_df['MEDICAL NOTES'].str.contains(' ear', case = False, na = False), 
                   'ear_infection'] = 1
    cleaned_df.loc[cleaned_df['ear_infection'] != 1, 
                   'ear_infection'] = 0
    cleaned_df.loc[cleaned_df['MEDICAL NOTES'].str.contains('tapeworm', case = False, na = False), 
                   'tapeworm'] = 1
    cleaned_df.loc[cleaned_df['tapeworm'] != 1, 
                   'tapeworm'] = 0
    cleaned_df.loc[cleaned_df['MEDICAL NOTES'].str.contains('infection', case = False, na = False), 
                   'general_infection'] = 1
    cleaned_df.loc[cleaned_df['general_infection'] != 1, 
                   'general_infection'] = 0
    cleaned_df.loc[cleaned_df['MEDICAL NOTES'].str.contains('demode', case = False, na = False), 
                   'demodex'] = 1
    cleaned_df.loc[cleaned_df['demodex'] != 1, 
                   'demodex'] = 0
    cleaned_df.loc[cleaned_df['MEDICAL NOTES'].str.contains('car', case = False, na = False) &
                   cleaned_df['MEDICAL NOTES'].str.contains('sick', case = False, na = False), 
                   'car_sick'] = 1
    cleaned_df.loc[cleaned_df['car_sick'] != 1, 
                   'car_sick'] = 0
    cleaned_df.loc[cleaned_df['MEDICAL NOTES'].str.contains('dog', case = False, na = False) &
                   cleaned_df['MEDICAL NOTES'].str.contains('park', case = False, na = False) &
                   cleaned_df['MEDICAL NOTES'].str.contains('not', case = False, na = False), 
                   'dog_park'] = 1
    cleaned_df.loc[cleaned_df['dog_park'] != 1, 
                   'dog_park'] = 0
    cleaned_df.loc[cleaned_df['MEDICAL NOTES'].str.contains('leg', case = False, na = False), 
                   'leg_issues'] = 1
    cleaned_df.loc[cleaned_df['leg_issues'] != 1, 
                   'leg_issues'] = 0
    cleaned_df.loc[cleaned_df['MEDICAL NOTES'].str.contains('anaplas', case = False, na = False), 
                   'anaplasmosis'] = 1
    cleaned_df.loc[cleaned_df['anaplasmosis'] != 1, 
                   'anaplasmosis'] = 0
    cleaned_df.loc[cleaned_df['MEDICAL NOTES'].str.contains('teeth', case = False, na = False) |
                   cleaned_df['MEDICAL NOTES'].str.contains('dental', case = False, na = False), 
                   'dental_issues'] = 1
    cleaned_df.loc[cleaned_df['dental_issues'] != 1, 
                   'dental_issues'] = 0
    cleaned_df.loc[cleaned_df['MEDICAL NOTES'].str.contains('weight', case = False, na = False), 
                   'weight_issues'] = 1
    cleaned_df.loc[cleaned_df['weight_issues'] != 1, 
                   'weight_issues'] = 0
    cleaned_df.loc[cleaned_df['MEDICAL NOTES'].str.contains('hair', case = False, na = False), 
                   'hair_loss'] = 1
    cleaned_df.loc[cleaned_df['hair_loss'] != 1, 
                   'hair_loss'] = 0
    cleaned_df.loc[cleaned_df['MEDICAL NOTES'].str.contains('treat', case = False, na = False) |
                   cleaned_df['MEDICAL NOTES'].str.contains('vaccin', case = False, na = False),  
                   'treated_vaccinated'] = 1
    cleaned_df.loc[cleaned_df['treated_vaccinated'] != 1, 
                   'treated_vaccinated'] = 0
    
    return cleaned_df

adopts_clean5 = clean_med_notes(adopts_clean4)

print('\nhas_med_issues\n', adopts_clean5['has_med_issues'].value_counts())
print('\ndiarrhea\n', adopts_clean5['diarrhea'].value_counts())
print('\nehrlichia\n', adopts_clean5['ehrlichia'].value_counts())
print('\nuri\n', adopts_clean5['uri'].value_counts())
print('\near_infection\n', adopts_clean5['ear_infection'].value_counts())
print('\ntapeworm\n', adopts_clean5['tapeworm'].value_counts())
print('\ngeneral_infection\n', adopts_clean5['general_infection'].value_counts())
print('\ndemodex\n', adopts_clean5['demodex'].value_counts())
print('\ncar_sick\n', adopts_clean5['car_sick'].value_counts())
print('\ndog_park\n', adopts_clean5['dog_park'].value_counts())
print('\nleg_issues\n', adopts_clean5['leg_issues'].value_counts())
print('\nanaplasmosis\n', adopts_clean5['anaplasmosis'].value_counts())
print('\ndental_issues\n', adopts_clean5['dental_issues'].value_counts())
print('\nweight_issues\n', adopts_clean5['weight_issues'].value_counts())
print('\nhair_loss\n', adopts_clean5['hair_loss'].value_counts())
print('\ntreated_vaccinated\n', adopts_clean5['treated_vaccinated'].value_counts())

#%% Clean up `SPAYED/NEUTERED` column
def clean_spay(df):
    
    cleaned_df = df.copy(deep = True)
    
    # Uncomment the below line if you want to make the dataframes easier to
    # work with, for testing purposes
#    cleaned_df = cleaned_df[['DOG NAME', 'MEDICAL NOTES', 'SPAYED/NEUTERED']]
    
    # Replace anything with 'Y' as 'Yes'; same for 'N' --> 'No'
    cleaned_df.loc[(cleaned_df['SPAYED/NEUTERED'].str.contains('y', case = False, na = False)) |
                    (cleaned_df['MEDICAL NOTES'].str.contains('s/n', case = False, na = False)), 
                    'spay_neuter'] = 1
    cleaned_df.loc[cleaned_df['SPAYED/NEUTERED'].str.contains('n', case = False, na = False), 'spay_neuter'] = 0
    
    return cleaned_df

adopts_clean6 = clean_spay(adopts_clean5)

print('\nSPAYED/NEUTERED\n', adopts_clean6['spay_neuter'].value_counts())

adopts_clean6 = clean_age(adopts_clean6)

#remove rows that have all Nan values
adopts_clean6.dropna(how="all")

# %% Write out our final results to a new CSV

try:
    adopts_clean6.to_csv('../Data/Cleaned_Adoption_List.csv', index = False)
except:
    adopts_clean6.to_csv('Data/Cleaned_Adoption_List.csv', index = False)
