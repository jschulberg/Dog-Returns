
import pandas as pd
import numpy as np
import sklearn
from matplotlib.colors import ListedColormap
from pandas.plotting import table
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from ScalingAndImputation import na_imputation, scale_arr
from AnalysisPrep import resample_data


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

# imputes values, resamples data, scales, and divides data into xtest/train and ytest/train.
def data_prep(df):
    data = na_imputation(df)
    y = data.iloc[:,-1]
    X = data.iloc[:,0:-1]

    # resample data
    xdata, ydata = resample_data(X.to_numpy(), y.to_numpy())

    # scale data
    xdata = scale_arr(xdata)

    # split into test and train sets
    xtrain, xtest, ytrain, ytest = train_test_split(xdata, ydata, test_size=0.2, random_state=0)

    return xtrain, xtest, ytrain, ytest

# calculate scores from each classifier and save into an image
def calc_scores(cf, classifier_type):
    df = pd.DataFrame(columns=["Classifier", "Accuracy", "Misclassification","Precision", "Recall", "Specificity", "F1"])

    # calculate scoring measures
    acc = (cf[0, 0] + cf[1, 1]) / np.sum(cf)
    mis = (cf[0, 1] + cf[1, 0]) / np.sum(cf)
    pre = cf[1, 1] / sum(cf[:, 1])
    recall = cf[1, 1] / sum(cf[1])
    spec = cf[0, 0] / sum(cf[0])
    f1 = 2 / (pre ** -1 + recall ** -1)
    l = [acc, mis, pre,recall, spec, f1]
    l2 = [round(i * 100, 2) for i in l]
    l2 = ['{:2}%'.format(elm) for elm in l2]
    l2.insert(0, classifier_type)
    df.loc[0] = l2

    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    table(ax, df)

    try:
        plt.savefig('Images/Scores' + classifier_type +'.png')

    except:
        plt.savefig('../Images/Scores' + classifier_type +'.png')

    plt.show()

    return


########### The following functions run the classifier on the prepped data set, and return a confusion matrix
########### and precision, recall, accuracy, and f1 scores

# Naive Bayes
def classifier_NB(xtrain, xtest, ytrain, ytest):
    gnb = GaussianNB()
    ypred = gnb.fit(xtrain, ytrain.flatten()).predict(xtest)

    cf = metrics.confusion_matrix(ytest.flatten(), ypred)
    ax= plt.subplot()
    sns.heatmap(cf, annot=True, fmt='g', ax=ax);

    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix - Naive Bayes');
    ax.xaxis.set_ticklabels(['Not Returned', 'Returned']); ax.yaxis.set_ticklabels(['Not Returned', 'Returned'])

    try:
        plt.savefig('Images/CM_NaiveBayes.png')

    except:
        plt.savefig('../Images/CM_NaiveBayes.png')

    plt.show()

    calc_scores(cf, "Naive Bayes")

    return


#KNN (might need to reduce sample size)
def classifier_KNN(xtrain, xtest, ytrain, ytest):
    kn = KNeighborsClassifier(n_neighbors=10)
    kn = kn.fit(xtrain, ytrain.flatten())
    ypred = kn.predict(xtest)
    cf = metrics.confusion_matrix(ytest.flatten(), ypred)
    ax= plt.subplot()
    sns.heatmap(cf, annot=True, fmt='g', ax=ax);

    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix - KNN');
    ax.xaxis.set_ticklabels(['Not Returned', 'Returned']);
    ax.yaxis.set_ticklabels(['Not Returned', 'Returned'])

    try:
        plt.savefig('Images/CM_KNN.png')

    except:
        plt.savefig('../Images/CM_KNN.png')

    plt.show()

    calc_scores(cf, "KNN")

    return

#Logistic Regression
def classifier_LR(xtrain, xtest, ytrain, ytest):
    lr = LogisticRegression(random_state=0, max_iter=120).fit(xtrain, ytrain.flatten())
    ypred = lr.predict(xtest)

    cf = metrics.confusion_matrix(ytest.flatten(), ypred)
    ax= plt.subplot()
    sns.heatmap(cf, annot=True, fmt='g', ax=ax);

    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix - Logistic Regression');
    ax.xaxis.set_ticklabels(['Not Returned', 'Returned']);
    ax.yaxis.set_ticklabels(['Not Returned', 'Returned'])

    try:
        plt.savefig('Images/CM_LR.png')

    except:
        plt.savefig('../Images/CM_LR.png')

    plt.show()

    calc_scores(cf, "Logistic Regression")

    return


#SVM (might need to reduce sample size)
def classifier_SVM(xtrain, xtest, ytrain, ytest):
    sv = svm.SVC()
    sv = sv.fit(xtrain, ytrain.flatten())
    ypred = sv.predict(xtest)

    cf = metrics.confusion_matrix(ytest.flatten(), ypred)
    ax= plt.subplot()
    sns.heatmap(cf, annot=True, fmt='g', ax=ax);

    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix - SVM');
    ax.xaxis.set_ticklabels(['Not Returned', 'Returned']);
    ax.yaxis.set_ticklabels(['Not Returned', 'Returned'])

    try:
        plt.savefig('Images/CM_SVM.png')

    except:
        plt.savefig('../Images/CM_SVM.png')

    plt.show()

    calc_scores(cf, "SVM")

    return

#Kernel SVM (might need to reduce sample size)
def classifier_KSVM(xtrain, xtest, ytrain, ytest):
    svk = svm.NuSVC(gamma = 'auto')
    svk = svk.fit(xtrain, ytrain.flatten())
    ypred = svk.predict(xtest)

    cf = metrics.confusion_matrix(ytest.flatten(), ypred)
    ax= plt.subplot()
    sns.heatmap(cf, annot=True, fmt='g', ax=ax);

    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix - Kernel SVM');
    ax.xaxis.set_ticklabels(['Not Returned', 'Returned']);
    ax.yaxis.set_ticklabels(['Not Returned', 'Returned'])

    try:
        plt.savefig('Images/CM_KSVM.png')

    except:
        plt.savefig('../Images/CM_KSVM.png')

    plt.show()

    calc_scores(cf, "Kernel SVM")

    return


#Neural networks
def classifier_NNet(xtrain, xtest, ytrain, ytest):
    mlp = MLPClassifier(hidden_layer_sizes=(20,10),max_iter=500)
    mlp.fit(xtrain,ytrain.flatten())
    ypred = mlp.predict(xtest)

    cf = metrics.confusion_matrix(ytest.flatten(), ypred)
    ax= plt.subplot()
    sns.heatmap(cf, annot=True, fmt='g', ax=ax);

    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix - Neural Networks');
    ax.xaxis.set_ticklabels(['Not Returned', 'Returned']);
    ax.yaxis.set_ticklabels(['Not Returned', 'Returned'])

    try:
        plt.savefig('Images/CM_NNet.png')

    except:
        plt.savefig('../Images/CM_NNet.png')

    plt.show()

    calc_scores(cf, "Neural Networks")

    return







xtrain, xtest, ytrain, ytest = data_prep(dogs_selected)
classifier_NB(xtrain, xtest, ytrain, ytest)