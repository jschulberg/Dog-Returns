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
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import dataframe_image as dfi

try:
    from ScalingAndImputation import na_imputation, scale_arr
    from AnalysisPrep import resample_data
except:
    from Code.ScalingAndImputation import na_imputation, scale_arr
    from Code.AnalysisPrep import resample_data
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import OneClassSVM
import dataframe_image as dfi
import prince
from sklearn.decomposition import PCA  # Used for principal component analysis
import re

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
dogs_selected = dogs_joined[['ID',
                             'SEX',
                             'multi_color',
                             'num_colors',
                             'MIX_BOOL',
                             'contains_black',
                             'contains_white',
                             'contains_yellow',
                             'contains_dark',
                             'WEIGHT2',
                             'Age at Adoption (days)',
                             'is_retriever',
                             'is_shepherd',
                             'is_terrier',
                             'is_husky',
                             'is_other_breed',
                             'num_behav_issues',
                             'new_this_week',
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

# imputes values, resamples data, scales, and divides data into xtest/train and ytest/train.
cols = dogs_selected.drop(columns=["ID", "returned"]).columns


def data_prep_PCA(df):
    print("Prepping data...")
    print("Imputing missing values...")
    data = na_imputation(df)
    y = data.iloc[:, -1]
    X = data.iloc[:, 0:-1]

    # resample data
    print("Resampling data...")
    xdata, ydata = resample_data(X.to_numpy(), y.to_numpy())

    # scale data
    print("Scaling data...")
    xdata = scale_arr(xdata)

    # Apply MCA (Multiple Correspondence Analysis), a version of PCA for
    # numerical and categorical variables
    mca = prince.MCA(
        n_components = 15,
        n_iter = 50,
        copy = True,
        check_input = True,
        engine = 'auto')

    xdata_mca = np.array(mca.fit(xdata).transform(xdata))
    xdata_pca = PCA(n_components = 10).fit_transform(xdata)

    X = xdata

    m, n = X.shape
    C = np.matmul(X.T, X) / m
    d = 2  # 4 dimensions
    V, s, nh = np.linalg.svd(C)

    V = V[:, :d]
    xdata_pca2 = np.dot(X, V)


    return xdata_pca2, ydata


#plots PCA image boundaries
def PCA_image_plot(xPCA, y):

    # split into test and train sets
    print("Splitting data into test/train sets...\n")
    xtrain, xtest, ytrain, ytest = train_test_split(xPCA, y, test_size=0.2, random_state=0)

    h = .02
    figure = plt.figure(figsize=(27, 9))
    i = 1
    x_min, x_max = xPCA[:, 0].min() - .5, xPCA[:, 0].max() + .5
    y_min, y_max = xPCA[:, 1].min() - .5, xPCA[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(1, 3, i)

    # Plot the training points
    ax.scatter(xtrain[:, 0], xtrain[:, 1], c=ytrain, cmap=cm_bright,
                   edgecolors='k')
    #Plot the testing points
    ax.scatter(xtest[:, 0], xtest[:, 1], c=ytest, cmap=cm_bright, alpha=0.6,
                   edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())


    names = ["Naive Bayes", "KNN k=1", "Logistic Regression", "SVM", "Kernel SVM", "Neural Networks"]
    classifiers = [GaussianNB(), KNeighborsClassifier(1), LogisticRegression(random_state=0, max_iter=130), \
                svm.SVC(), svm.NuSVC(gamma = 'auto'), MLPClassifier(hidden_layer_sizes=(20,10),max_iter=500)]


    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        print("Running...")
        ax = plt.subplot(2, 3, i)
        clf.fit(xtrain, ytrain)
        score = clf.score(xtest, ytest)
        r = np.c_[xx.ravel(), yy.ravel()]
    # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

            # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot the training points
        ax.scatter(xtrain[:, 0], xtrain[:, 1], c=ytrain, cmap=cm_bright,
                       edgecolors='k')
        # Plot the testing points
        ax.scatter(xtest[:, 0], xtest[:, 1], c=ytest, cmap=cm_bright,
                       edgecolors='k', alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

    plt.tight_layout()
    try:
        plt.savefig('Images/PCAPlots.png')

    except:
        plt.savefig('../Images/PCAPlots.png')

    plt.show()

    return

x, y = data_prep_PCA(dogs_selected)
PCA_image_plot(x,y)