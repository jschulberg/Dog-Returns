
import pandas as pd
import numpy as np
import sklearn
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from MissingValueImputation import na_imputation
#from AnalysisPrep import resample_data


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

data = na_imputation(dogs_selected)
#resample
#scale

y = data.iloc[:,-1]
X = data.iloc[:,0:-1]
xtrain, xtest, ytrain, ytest = train_test_split(X.to_numpy(), y.to_numpy(), test_size=0.2, random_state=0)


# Naive Bayes

gnb = GaussianNB()
ypred = gnb.fit(xtrain, ytrain.flatten()).predict(xtest)

cf = metrics.confusion_matrix(ytest.flatten(), ypred)
ax= plt.subplot()
sns.heatmap(cf, annot=True, fmt='g', ax=ax);

ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['0', '1'])

plt.show()
df = pd.DataFrame(columns = ["Number", "Precision", "Recall", "F1"])

for i in range(0,2):
    p = cf[i,i]/sum(cf[:,i])
    r = cf[i,i] / sum(cf[i])
    f1 = 2/(p**-1 + r**-1)
    l = [i,p,r,f1]
    df.loc[len(df.index)] = l

print(df)


#KNN (might need to reduce sample size)

kn = KNeighborsClassifier(n_neighbors=10)
kn = kn.fit(xtrain, ytrain.flatten())
ypred = kn.predict(xtest)
cf = metrics.confusion_matrix(ytest.flatten(), ypred)
ax= plt.subplot()
sns.heatmap(cf, annot=True, fmt='g', ax=ax);

ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['0', '1'])

plt.show()
df = pd.DataFrame(columns = ["Number", "Precision", "Recall", "F1"])

for i in range(0,2):
    p = cf[i,i]/sum(cf[:,i])
    r = cf[i,i] / sum(cf[i])
    f1 = 2/(p**-1 + r**-1)
    l = [i,p,r,f1]
    df.loc[len(df.index)] = l

print(df)

#Logistic Regression

lr = LogisticRegression(random_state=0).fit(xtrain, ytrain.flatten())
ypred = lr.predict(xtest)

cf = metrics.confusion_matrix(ytest.flatten(), ypred)
ax= plt.subplot()
sns.heatmap(cf, annot=True, fmt='g', ax=ax);

ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['0', '1'])

plt.show()
df = pd.DataFrame(columns = ["Number", "Precision", "Recall", "F1"])

for i in range(0,2):
    p = cf[i,i]/sum(cf[:,i])
    r = cf[i,i] / sum(cf[i])
    f1 = 2/(p**-1 + r**-1)
    l = [i,p,r,f1]
    df.loc[len(df.index)] = l

print(df)


#SVM (might need to reduce sample size)

sv = svm.SVC()
sv = sv.fit(xtrain, ytrain.flatten())
ypred = sv.predict(xtest)

cf = metrics.confusion_matrix(ytest.flatten(), ypred)
ax= plt.subplot()
sns.heatmap(cf, annot=True, fmt='g', ax=ax);

ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['0', '1'])

plt.show()
df = pd.DataFrame(columns = ["Number", "Precision", "Recall", "F1"])

for i in range(0,2):
    p = cf[i,i]/sum(cf[:,i])
    r = cf[i,i] / sum(cf[i])
    f1 = 2/(p**-1 + r**-1)
    l = [i,p,r,f1]
    df.loc[len(df.index)] = l

print(df)

#Kernel SVM (might need to reduce sample size)

svk = svm.NuSVC(gamma = 'auto')
svk = svk.fit(xtrain, ytrain.flatten())
ypred = svk.predict(xtest)

cf = metrics.confusion_matrix(ytest.flatten(), ypred)
ax= plt.subplot()
sns.heatmap(cf, annot=True, fmt='g', ax=ax);

ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['0', '1'])

plt.show()
df = pd.DataFrame(columns = ["Number", "Precision", "Recall", "F1"])

for i in range(0,2):
    p = cf[i,i]/sum(cf[:,i])
    r = cf[i,i] / sum(cf[i])
    f1 = 2/(p**-1 + r**-1)
    l = [i,p,r,f1]
    df.loc[len(df.index)] = l

print(df)


#Neural networks

mlp = MLPClassifier(hidden_layer_sizes=(20,10),max_iter=500)
mlp.fit(xtrain,ytrain.flatten())
ypred = mlp.predict(xtest)

cf = metrics.confusion_matrix(ytest.flatten(), ypred)
ax= plt.subplot()
sns.heatmap(cf, annot=True, fmt='g', ax=ax);

ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['0', '1'])

plt.show()
df = pd.DataFrame(columns = ["Number", "Precision", "Recall", "F1"])

for i in range(0,2):
    p = cf[i,i]/sum(cf[:,i])
    r = cf[i,i] / sum(cf[i])
    f1 = 2/(p**-1 + r**-1)
    l = [i,p,r,f1]
    df.loc[len(df.index)] = l

print(df)