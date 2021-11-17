# Final Project Proposal: Lucky Dog Animal Rescue Returns Classification

## Problem Statement
For the final project, we will be analyzing a dataset from Lucky Dog Animal Rescue (LDAR), a nonprofit dedicated to rescuing homeless, neglected, and abandoned animals from euthanasia in kill shelters and getting them adopted into their forever homes. LDAR educates the community and all pet parents on responsible pet parenting, including the importance of spay/neuter, obedience training, and good nutrition.

Successful adoptions of cats and dogs depend on a variety of factors, both on the parts of the pet and of the family. LDAR helps bridge the gap, building strong relationships between the two. LDAR rescues hundreds of animals every year, provides them with loving temporary care, and finds them well-matched, carefully screened forever homes.

One of the struggles that Lucky Dog runs into is pets being returned after adoption. Even though they adopt out ~2000 dogs per year, about 10% get returned for a variety of factors. LDAR assiduously tracks information on all their adoptions and returns. With that, we will use various analytical methods learned in Computational Data Analysis to help Lucky Dog Animal Rescue predict whether or not a dog that they adopt out will be returned. 

## Data Source
The project will be overseen by Julie Brooks, Program Manager for Volunteers and Data Integrity at LDAR. She will provide the data needed for this analysis to we. Data will be provided using two official sources from LDAR, both of which contain records over the past decade, and can be easily linked by a unique ID field for each dog represented in each dataset. The two main data sources are:

**Dog List** | A list of every dog that has been adopted out by LDAR over the past 10 years. This list includes a variety of features describing each dog, where each row corresponds to a dog being adopted out, and each column represents a different attribute related to that dog. This dataset is composed of 10 spreadsheets, one for every year over the past decade. Each spreadsheet has records of ~2000 adoptions per year. The attributes in this dataset are as follows:


| Field            | Description                                                                                                                                                                                                    |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dog Name         | Name of the adopted dog                                                                                                                                                                                        |
| ID               | Unique ID corresponding to that dog                                                                                                                                                                            |
| Link             | Link to dog’s profile on LDAR site                                                                                                                                                                             |
| Foster/Boarding  | Type of adoption for the dog (short vs. long-term)                                                                                                                                                             |
| Sex              | Gender of dog                                                                                                                                                                                                  |
| Age              | Estimated age (sometimes a range if actual age not known) of dog at time of adoption                                                                                                                           |
| Weight           | Estimated weight of dog at time of adoption                                                                                                                                                                    |
| Breed Mixes      | Type of dog                                                                                                                                                                                                    |
| Color            | Color of dog                                                                                                                                                                                                   |
| Behavioral Notes | Free text field describing the behavior of the dog. There’s some consistency in entries depending on the individual entering this field, along with some key words to describe the dogs behavior around others |
| Dogs in Home     | Number of other dogs in adoptee’s house, if any at all                                                                                                                                                         |
| Cats in Home     | Number of cats in adoptee’s house, if any at all                                                                                                                                                               |
| Kids             | Number of kids in adoptee’s house, if any at all                                                                                                                                                               |
| BS/W             | Indicator for whether or not the dog is part pitbull or other ‘bully’ breed                                                                                                                                    |
| Medical Notes    | Free text field describing any health conditions of the dog, if any exist at all                                                                                                                               |
| Transport Date   | Date of adoption                                                                                                                                                                                               |


**Returns List** | A list of every dog that’s been returned after being adopted out by LDAR. This list also includes a variety of features describing each dog, where each row corresponds to a dog being adopted out, and each column represents a different attribute related to that dog. This dataset is composed of 10 spreadsheets, one for every year over the past decade. Each spreadsheet has records of ~200 returns per year.

| Field                 | Description                                                                                                                                                                                              |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dog Name              | Name of the adopted dog                                                                                                                                                                                  |
| ID                    | Unique ID corresponding to that dog                                                                                                                                                                      |
| Dog Info              | Free text field with general information on the dog                                                                                                                                                      |
| Reason for Return     | Free text field describing the reason the dog was returned by the adopters. There’s some consistency in this field, but it generally depends on the person entering the information into the spreadsheet |
| Behavior with Dogs    | Free text field describing the dog’s general behavior around other dogs                                                                                                                                  |
| Behavior with Kids    | Free text field describing the dog’s general behavior around kids                                                                                                                                        |
| Behavior with Cats    | Free text field describing the dog’s general behavior around cats                                                                                                                                        |
| Energy Level          | General energy of the dog                                                                                                                                                                                |
| Socialization/Daycare | Categorical field for whether the dog was in a daytime program for dogs                                                                                                                                  |
| Vetting               | Categorical field for whether the dog was being taken to the vet                                                                                                                                         |
| Date of Adoption      | Date dog was adopted                                                                                                                                                                                     |
| Previous Return?      | Boolean for whether the dog has been returned in the past                                                                                                                                                |
| Previous Return Info  | Free text field describing why the dog had been returned in the past                                                                                                                                     |
| Date of Return        | Date dog was returned                                                                                                                                                                                    |
| Type                  | Boolean for dog vs. puppy                                                                                                                                                                                |


## Methodology
The goal of this project is to help keep dogs adopted by identifying dogs that are at risk of being returned. We will do this by building classifier models that classify the adopted dogs on whether they are returned vs. stay adopted, using the historical data we have about each dog.

### Data Preparation
To begin, we will have to process the data as it is not fully prepared for data analysis. Currently, all data is kept in a different spreadsheet by year (i.e. adoptions in 2021 are in the ‘Dog List 2021’; so we’ll have to programmatically combine all of our datasets. On top of that, the Dog and Returns Lists are kept as two separate data sources, but share a linking variable ‘ID’ for dogs that were returned. We’ll have to use a Left Join to find dogs that were returned; for any dogs which don’t have a match in the join, we’ll flag those as ‘Not Returned’.

There are also some issues with the features provided in the data. For example, the feature for ‘age’ is not standardized (written as weeks, months, or years) and is sometimes even written as an interval if the dog’s age is unknown, so we will need to standardize those features using one age marker. Additionally, some factors are described via text. For example, high energy dogs have the ‘description’ “walks not enough, needs play” - so we will attempt to identify those characteristics by scraping the text and then marking them as binary variables. We will also need to read in features like ‘breed’ as factor variables, in order for the model to read them correctly.  Following this initial preparation, we will standardize the numerical data using a python data scaling package to ensure certain features are not biased, such as weight vs. age. 

### Analysis
After preparing the data, we will analyze the data using the various classification models that we have learned so far in class including, but not limited to, Naïve Bayes, K Nearest Neighbors, SVM, Logistic Regression, and Neural Networks. We will split the data into training, test, and validation sets, as we are using multiple models. By testing each of these methods, we hope to identify which one is most successful in classifying the adopted dogs as returned vs. not-returned. 

## Evaluation and Final Results
To evaluate our models, we will look at their classification/misclassification rates using a confusion matrix, as well as their precision, recall, and F1 scores. These metrics are essential in judging the outcome of a classification model, which is why we will use it to evaluate ours. We will compare the results of our test and validation sets in order to determine which model is the most accurate. We’ll also take note of the performance efficiency of each model.

If our models are accurate, they can be integrated into Lucky Dog’s systems to classify new dogs that come into LDAR. If a dog is classified into the return group, Lucky Dog can ensure that the dog is well matched with its new adopter, and reach out to the adopter to provide additional support to prevent the dog from being returned. We plan to construct this project so  that Lucky Dog’s data team can continue to use it, helping to place more rescued dogs in loving homes and keep them there.
