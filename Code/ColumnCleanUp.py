import pandas as pd
import numpy as np
import os
import re

adopts = pd.read_csv("../Data/Master_Adoption_List.csv")
returns = pd.read_csv("../Data/Master_Returns_List.csv")

# %%
#adoption_original = pd.read_csv('Data/Master_Adoption_List.csv')


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
    cleaned_df = df.copy()

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
        cleaned_df['COLOR'].str.lower().str.lower().str.contains('screen').replace(np.nan, False), 'COLOR_FIXED'] = \
    cleaned_df['BREED MIXES']
    cleaned_df.loc[
        cleaned_df['COLOR'].str.lower().str.lower().str.contains('walks alone').replace(np.nan, False), 'COLOR_FIXED'] = \
    cleaned_df['BREED MIXES']
    cleaned_df.loc[cleaned_df['COLOR'].str.lower().str.lower().str.contains('month old puppy').replace(np.nan,
                                                                                                       False), 'COLOR_FIXED'] = \
    cleaned_df['BREED MIXES']
    cleaned_df.loc[cleaned_df['COLOR'].str.lower().str.lower().str.contains('new this week').replace(np.nan,
                                                                                                     False), 'COLOR_FIXED'] = \
    cleaned_df['BREED MIXES']
    cleaned_df.loc[
        cleaned_df['COLOR'].str.lower().str.lower().str.contains('kids').replace(np.nan, False), 'COLOR_FIXED'] = \
    cleaned_df['BREED MIXES']
    cleaned_df.loc[
        cleaned_df['COLOR'].str.lower().str.lower().str.contains('energy').replace(np.nan, False), 'COLOR_FIXED'] = \
    cleaned_df['BREED MIXES']
    cleaned_df.loc[cleaned_df['COLOR'].str.lower().str.contains('HW').replace(np.nan, False), 'COLOR_FIXED'] = \
    cleaned_df['BREED MIXES']
    cleaned_df.loc[
        cleaned_df['COLOR'].str.lower().str.lower().str.contains('nervous').replace(np.nan, False), 'COLOR_FIXED'] = \
    cleaned_df['BREED MIXES']
    cleaned_df.loc[
        cleaned_df['COLOR'].str.lower().str.lower().str.contains('retriever').replace(np.nan, False), 'COLOR_FIXED'] = \
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
    tri_color_bool = (cleaned_df['COLOR_FIXED'].str.contains('tri') & cleaned_df['COLOR_FIXED'].str.contains('color')) \
        .replace(np.nan, False)
    multi_color_bool = (cleaned_df['COLOR_FIXED'].str.contains(',')).replace(np.nan, False)

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
    cleaned_df.loc[~(cleaned_df['COLOR_FIXED'].str.contains('black').replace(np.nan, False)) & ~(
        cleaned_df['COLOR_FIXED'].isna()), 'contains_black'] = 0
    cleaned_df.loc[
        (cleaned_df['COLOR_FIXED'].str.contains('white')) & ~(cleaned_df['COLOR_FIXED'].isna()), 'contains_white'] = 1
    cleaned_df.loc[~(cleaned_df['COLOR_FIXED'].str.contains('white').replace(np.nan, False)) & ~(
        cleaned_df['COLOR_FIXED'].isna()), 'contains_white'] = 0
    # Create a boolean for tan/yellow/golden
    yellow_bool = (cleaned_df['COLOR_FIXED'].str.contains('tan')) | (
        cleaned_df['COLOR_FIXED'].str.contains('yellow')) | (cleaned_df['COLOR_FIXED'].str.contains('golden'))
    cleaned_df.loc[(yellow_bool) & ~(cleaned_df['COLOR_FIXED'].isna()), 'contains_yellow'] = 1
    cleaned_df.loc[~(yellow_bool) & ~(cleaned_df['COLOR_FIXED'].isna()), 'contains_yellow'] = 0

    return cleaned_df


def clean_sex(df):

    # update adoption list gender column to Male or Female and remove typos/misspellings
    cleaned_df = df.copy()
    df_obj = cleaned_df.select_dtypes(['object'])
    cleaned_df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    cleaned_df.loc[cleaned_df["SEX"].str.contains("f", case = False, na = False, regex=False), 'SEX'] = "Female"
    cleaned_df.loc[cleaned_df["SEX"].str.startswith("m", na = False), 'SEX'] = "Male"
    cleaned_df.loc[cleaned_df["SEX"].str.startswith("M", na = False), 'SEX'] = "Male"

    return cleaned_df


def clean_weight(df):
    # update adoption list weight column
    cleaned_df = df.copy(deep = True)
    cleaned_df["WEIGHT2"] = cleaned_df.loc[~cleaned_df["WEIGHT"].str.contains("/", case=False, na=False, regex=False), 'WEIGHT']
    cleaned_df["WEIGHT2"] = cleaned_df["WEIGHT2"].str.extract("([-+]?\d*\.\d+|[-+]?\d+)").astype(float)
    #cleaned_df.drop(["WEIGHT"])

    #cleaned_df.rename({"WEIGHT2: WEIGHT"})

    return cleaned_df


def clean_age(df):
    # update adoption list age column, provide age in days
    cleaned_df = df.copy()

    cleaned_df['DOB2'] = cleaned_df['AGE'].str.extract(r"(\d{1,2}[/. ](?:\d{1,2}|January|Jan)[/. ]\d{2}(?:\d{2})?)")
    cleaned_df['Date of Birth'] = cleaned_df['Date of Birth'].fillna(cleaned_df.pop('DOB2'))

    print(cleaned_df.head())

    cleaned_df['Date of Birth'] = pd.to_datetime(cleaned_df['Date of Birth'], errors='coerce')
    cleaned_df['Adopted On'] = pd.to_datetime(cleaned_df['Adopted On'], errors='coerce')
    cleaned_df["Age at Adoption (days)"] = (cleaned_df['Adopted On'] - cleaned_df['Date of Birth']).dt.days

    cleaned_df.loc[cleaned_df["Age at Adoption (days)"].isnull(), "Age at Adoption (days)"] = (pd.to_datetime("today") - cleaned_df['Date of Birth']).dt.days


    return cleaned_df


dc = clean_weight(adopts)
print(dc.head())


# # Run the function
# if 'COLOR' in adoption_original.columns:
#     adoption_cleaned = clean_color(adoption_original)
#
#     # Print out our new results
#     print(adoption_cleaned['COLOR'].value_counts())
#     print(adoption_cleaned['COLOR_FIXED'].value_counts())
#
# else:
#     print("WARNING! Unable to find the 'COLOR' column in the original dataset.",
#           "Make sure the column is in the dataset and spelled correctly.")
#
# # %% Write out our final results to a new CSV
# adoption_cleaned.to_csv('Data/Cleaned_Adoption_List.csv', index=False)