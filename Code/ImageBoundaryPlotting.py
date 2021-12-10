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
from sklearn.svm import OneClassSVM
import dataframe_image as dfi


def PCA_image_plot(xPCA,xtrain, xtest, ytrain, ytest):

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


    names = ["Naive Bayes", "Logistic Regression", "KNN k=3"]
    classifiers = [GaussianNB(), KNeighborsClassifier(10), LogisticRegression(random_state=0, max_iter=130), \
                svm.SVC(), svm.NuSVC(gamma = 'auto'), MLPClassifier(hidden_layer_sizes=(20,10),max_iter=500)]


    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(2, 3, i)
        clf.fit(xtrain, ytrain)
        score = clf.score(xtest, ytest)

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
    plt.show()

    return
