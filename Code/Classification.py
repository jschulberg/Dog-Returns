import pandas as pd
import numpy as np
import sklearn
from matplotlib.colors import ListedColormap
from pandas.plotting import table
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn import svm
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
from sklearn.decomposition import PCA # Used for principal component analysis
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
cols = dogs_selected.drop(columns = ["ID", "returned"]).columns

def data_prep(df):
    print("Prepping data...")
    print("Imputing missing values...")
    data = na_imputation(df)
    y = data.iloc[:,-1]
    X = data.iloc[:,0:-1]

    # resample data
    print("Resampling data...")
    xdata, ydata = resample_data(X.to_numpy(), y.to_numpy())

    # scale data
    print("Scaling data...")
    xdata = scale_arr(xdata)
    
    # Apply MCA (Multiple Correspondence Analysis), a version of PCA for
    # numerical and categorical variables
    # mca = prince.MCA(
    #     n_components = 15,
    #     n_iter = 50,
    #     copy = True,
    #     check_input = True,
    #     engine = 'auto')
    
    # xdata_mca = np.array(mca.fit(xdata).transform(xdata))
    # xdata_pca = PCA(n_components = 10).fit_transform(xdata)

    # split into test and train sets
    print("Splitting data into test/train sets...\n")
    xtrain, xtest, ytrain, ytest = train_test_split(xdata, ydata, test_size=0.2)

    return xtrain, xtest, ytrain, ytest

# calculate scores from each classifier and save into an image
def calc_scores(cf, classifier_type, *df2):
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

    # try:
    #     dfi.export(df, 'Images/Scores' + classifier_type +'.png')

    # except:
    #     dfi.export(df, '../Images/Scores' + classifier_type +'.png')


    # print(df)
    print(f"Accuracy: {df['Accuracy'].values[0]}\n")


    return df


########### The following functions run the classifier on the prepped data set, and return a confusion matrix
########### and precision, recall, accuracy, and f1 scores

# Naive Bayes
def classifier_NB(xtrain, xtest, ytrain, ytest):
    print(f"Running Naive Bayes...")
    gnb = GaussianNB()
    
    # Attempt to tune the Gaussian Naive Bayes model
    params_NB = {'var_smoothing': np.logspace(0,-9, num = 100)}
    
    gs_NB = GridSearchCV(estimator = gnb, 
                     param_grid=params_NB, 
                     cv = 5, # Use 5 folds
                     verbose = 1, 
                     scoring = 'accuracy') 
    gs_NB.fit(xtrain, ytrain)
        

    ypred = gs_NB.fit(xtrain, ytrain.flatten()).predict(xtest)
    # ypred = gnb.fit(xtrain, ytrain.flatten()).predict(xtest)

    cf = metrics.confusion_matrix(ytest.flatten(), ypred)
    ax= plt.subplot()
    sns.heatmap(cf, 
                annot=True, 
                fmt='g', 
                cbar = False,
                cmap = 'Purples',
                ax=ax);

    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix - Naive Bayes');
    ax.xaxis.set_ticklabels(['Not Returned', 'Returned']); ax.yaxis.set_ticklabels(['Not Returned', 'Returned'])

    try:
        plt.savefig('Images/CM_NaiveBayes.png')

    except:
        plt.savefig('../Images/CM_NaiveBayes.png')

    plt.show()

    c = calc_scores(cf, "Naive Bayes")
    
    return c


#KNN (might need to reduce sample size)
def classifier_KNN(xtrain, xtest, ytrain, ytest):
    print(f"Running KNN...")
    
    # Run through up to 15 neighbors for KNN (Only odd values)
    accuracy_vals_knn = []
    k_vals = [k for k in range(1, 16) if k%2 - 1 == 0]
    for k in k_vals:
        print("k =", k)
        kn = KNeighborsClassifier(n_neighbors = k)
        kn = kn.fit(xtrain, ytrain.flatten())
        ypred = kn.predict(xtest)
        cf = metrics.confusion_matrix(ytest.flatten(), ypred)
        c = calc_scores(cf, "KNN")
        # Append the accuracy to our list, but remove the '%' symbol
        accuracy_vals_knn.append(float(re.sub('%', '', c['Accuracy'].values[0])))
        
    # Let's plot our accuracy points
    plt.figure(figsize = (8,8))
    
    plt.plot(k_vals,
             accuracy_vals_knn,
             linewidth = 4,
             marker = 'o',
             markersize = 10,
             c = 'Slateblue')
    
    # Set the y-axis
    plt.ylim([min(accuracy_vals_knn) - 5, max(accuracy_vals_knn) + 5])
    plt.xlabel('# of Neighbors', fontsize = 14)
    plt.ylabel('Accuracy', fontsize = 14)
    plt.title('KNN: Accuracy of Different K Values', fontsize = 18)
    
    plt.savefig('Images/KNN_K_Values_Accuracy.png', bbox_inches='tight')
    plt.show()     
    
    # Now run knn with our best k value
    print(f"Best k-value for KNN @ k = {k_vals[np.where(min(accuracy_vals_knn))[0][0]]}")
    kn = KNeighborsClassifier(n_neighbors = k_vals[np.where(min(accuracy_vals_knn))[0][0]])
    kn = kn.fit(xtrain, ytrain.flatten())
    ypred = kn.predict(xtest)
    cf = metrics.confusion_matrix(ytest.flatten(), ypred)
    c = calc_scores(cf, "KNN")

    ax= plt.subplot()
    sns.heatmap(cf, 
                annot=True, 
                fmt='g', 
                cmap = 'Purples',
                cbar = False,                
                ax=ax);
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

    
    return c

#Logistic Regression
def classifier_LR(xtrain, xtest, ytrain, ytest):
    print(f"Running Logistic Regression...")
    
    # Let's try out various C values for the 5 solvers we can use
    solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    
    # Before iterating through them, let's initialize an empty dataframe
    log_reg_df = pd.DataFrame(columns = ['solver', 'C', 'accuracy'])
    
    for solver in solvers:
        print("-"*50)
        print("Solver:", solver)
        print("-"*50)
        
        accuracy_vals_logreg = []
        # c_vals = [c for c in range(1, 11) if c%2 == 0]
        c_vals = [.001, .01, .05, .1, .25, .5, 1, 2, 5, 10, 100, 1000]

        for c in c_vals:
            print("C =", c)
            lr = LogisticRegression(solver = solver,
                                    C = c,
                                    max_iter = 200).fit(xtrain, ytrain.flatten())
            ypred = lr.predict(xtest)
            cf = metrics.confusion_matrix(ytest.flatten(), ypred)
            c = calc_scores(cf, "Logistic Regression")
            # Append the accuracy to our list, but remove the '%' symbol
            accuracy_vals_logreg.append(float(re.sub('%', '', c['Accuracy'].values[0])))
            
        # Convert these into a dataframe
        temp_df = pd.DataFrame(zip(c_vals, accuracy_vals_logreg), columns = ['C', 'accuracy'])
        temp_df['solver'] = solver
        log_reg_df = pd.concat([log_reg_df, temp_df])
            
        # Let's plot our accuracy points for each solver
        plt.figure(figsize = (8,8))
        
        plt.plot(np.log10(c_vals),
                 accuracy_vals_logreg,
                 linewidth = 4,
                 marker = 'o',
                 markersize = 10,
                 c = 'Slateblue')
        
        # Set the y-axis
        plt.ylim([min(accuracy_vals_logreg) - 5, max(accuracy_vals_logreg) + 5])
        plt.xlabel('C (Inverse of Regularization Strength)', fontsize = 14)
        plt.ylabel('Accuracy', fontsize = 14)
        plt.title(f'Logistic Regression: Accuracy of Different C Values\nSolver: {solver}', fontsize = 18)
        
        plt.savefig(f'Images/Logistic_Regression_{solver}_C_Values_Accuracy.png', bbox_inches='tight')
        plt.show()     
    
    
    # Pull out the parameters from our best logistic regression classifier
    log_reg_best = log_reg_df.loc[log_reg_df['accuracy'] == max(log_reg_df['accuracy']), :].sample(1)
    print("Best logistic regression parameters:\n", log_reg_best)
    lr_best = LogisticRegression(solver = log_reg_best.iloc[0, 0],
                                 C = log_reg_best.iloc[0, 1],
                                 max_iter = 200).fit(xtrain, ytrain.flatten())
    ypred = lr_best.predict(xtest)

    cf = metrics.confusion_matrix(ytest.flatten(), ypred)
    ax= plt.subplot()
    sns.heatmap(cf, 
                annot=True, 
                fmt='g', 
                cmap = 'Purples',
                cbar = False,                
                ax=ax);
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

    c = calc_scores(cf, "Logistic Regression")
    
    
    # Let's take a look at the most important features

    return c


#SVM (might need to reduce sample size)
def classifier_SVM(xtrain, xtest, ytrain, ytest):
    print(f"Running SVM...")
    sv = svm.SVC()
    sv = sv.fit(xtrain, ytrain.flatten())
    ypred = sv.predict(xtest)

    cf = metrics.confusion_matrix(ytest.flatten(), ypred)
    ax= plt.subplot()
    sns.heatmap(cf, 
                annot=True, 
                fmt='g', 
                cmap = 'Purples',
                cbar = False,                
                ax=ax);
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

    c = calc_scores(cf, "SVM")

    return c

#Kernel SVM (might need to reduce sample size)
def classifier_KSVM(xtrain, xtest, ytrain, ytest):
    print(f"Running Kernel SVM...")
    svk = svm.NuSVC(gamma = 'auto')
    svk = svk.fit(xtrain, ytrain.flatten())
    ypred = svk.predict(xtest)

    cf = metrics.confusion_matrix(ytest.flatten(), ypred)
    ax= plt.subplot()
    sns.heatmap(cf, 
                annot=True, 
                fmt='g', 
                cmap = 'Purples',
                cbar = False,                
                ax=ax);

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

    c = calc_scores(cf, "Kernel SVM")

    return c


#Neural networks
def classifier_NNet(xtrain, xtest, ytrain, ytest):
    print(f"Running Neural Network...")
    mlp = MLPClassifier(hidden_layer_sizes=(20,10),max_iter=500)
    mlp.fit(xtrain,ytrain.flatten())
    ypred = mlp.predict(xtest)

    cf = metrics.confusion_matrix(ytest.flatten(), ypred)
    ax= plt.subplot()
    sns.heatmap(cf, 
                annot=True, 
                fmt='g', 
                cmap = 'Purples',
                cbar = False,                
                ax=ax);
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

    c = calc_scores(cf, "Neural Networks")

    return c

# Decision Tree
def classifier_tree(xtrain, xtest, ytrain, ytest, cols):
    print(f"Running Decision Tree...")
    dtc = DecisionTreeClassifier(random_state = 42)
    dtc.fit(xtrain, ytrain)
    ypred = dtc.predict(xtest)

    plt.figure(figsize=(24, 24))
    tree.plot_tree(dtc, 
                   feature_names = cols,
                   class_names = ['Not Returned', 'Returned'],
                   filled = True, 
                   rounded = True,
                   impurity = False,
                   label = 'root',
                   max_depth = 3, 
                   fontsize = 20)
    plt.title('Abbreviated Decision Tree (Depth = 2)', size = 48)

    try:
        plt.savefig('Images/TreePlot.png')

    except:
        plt.savefig('../Images/TreePlot.png')

    plt.show()

    cf = metrics.confusion_matrix(ytest.flatten(), ypred)
    ax = plt.subplot()
    sns.heatmap(cf, 
                annot=True, 
                fmt='g', 
                cmap = 'Purples',
                cbar = False,                
                ax=ax);

    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix - Decision Tree');
    ax.xaxis.set_ticklabels(['Not Returned', 'Returned']);
    ax.yaxis.set_ticklabels(['Not Returned', 'Returned'])

    try:
        plt.savefig('Images/CM_tree.png')

    except:
        plt.savefig('../Images/CM_tree.png')

    plt.show()

    print('*************** Decision Tree Summary ***************')
    print('Classes: ', dtc.classes_)
    print('Tree Depth: ', dtc.tree_.max_depth)
    print('# of leaves: ', dtc.tree_.n_leaves)
    print('# of features: ', dtc.n_features_)
    c = calc_scores(cf, "Decision Tree")
    print('--------------------------------------------------------\n\n')
    
    
    # Get feature importance, which is dependent on the number of instances
    # that a feature appears in the trees
    tree_importances = pd.DataFrame(zip(cols.tolist(), dtc.feature_importances_.tolist()), 
                                      columns = ['feature', 'importance'])
    
    tree_importances_top10 = tree_importances.sort_values('importance', ascending = False).head(10)
    
    plt.figure(figsize = (8,8))

    plt.bar(tree_importances_top10['feature'],
             # Convert the accuracy values from a string to a float and remove
             # the '%' symbol at the end of it and sort the results in descending
             # order
             sorted(tree_importances_top10['importance'], reverse = True),
             color = 'Slateblue')
    
    # Set the y-axis
    plt.ylim([min(tree_importances_top10['importance']) - .05, 
              max(tree_importances_top10['importance']) + .05])
    # Tilt the x-labels
    plt.xticks(rotation = 45)
    plt.xlabel('Feature', fontsize = 14)
    plt.ylabel('Importance (Based on Impurity)', fontsize = 14)
    plt.title('Feature Importance on Decision Trees', fontsize = 18)
    
    plt.savefig('Images/Decision_Tree_Feature_Importance.png', bbox_inches='tight')
    plt.show() 
    

    return c


# Random Forest
def classifier_RF(xtrain, xtest, ytrain, ytest):
    print(f"Running Random Forest...")
    dtc = DecisionTreeClassifier(random_state=24)
    dtc.fit(xtrain, ytrain)
    rf_misc = []
    for i in range(1, 200):
        # Progress check
        if (i-1)%50 == 0:
            print(f"Running Random Forest Classifier for Number of Trees {i}:{i+49}...")
            
        rf = RandomForestClassifier(n_estimators = i,random_state=24)
        rf.fit(xtrain, ytrain)
        rf_misc.append(1 - rf.score(xtest, ytest))

    min_value = min(rf_misc)
    min_index = rf_misc.index(min_value)
    rf_best = RandomForestClassifier(n_estimators=min_index, random_state=24)
    rf_best.fit(xtrain, ytrain)
    ypred = rf_best.predict(xtest)

    dtc_misc = 1 - dtc.score(xtest, ytest)

    # Plot error rate vs. # of trees for Random Forest
    plt.plot(range(1, 200), 
             rf_misc, # error rate is 1 - accuracy rate
             color = 'slateblue',
             alpha = .8)
    
    # Plot of our consistent Decision Tree error
    # across the "number of trees" from earlier
    plt.hlines(y = dtc_misc, 
               xmin = 0, 
               xmax = 200,
               alpha = .8,
               colors = 'Gray', 
               lw = 2, 
               label = 'CART Misclassification Rate')
    
    # Big dot on the max value
    plt.scatter(min_index,
                min_value, 
                c = 'slateblue',
                s = 50)
    
    # Plot text for max accuracy point(s) 
    # a bit to the top-right of the point
    plt.text(min_index + 3, 
                min_value + min_value/15,
            s = f"Min Error: {round(min_value, 4)}\nRF Trees: {min_index}",
            fontsize = 8)
    
    plt.title('Error Rate of Random Forest Classifier\nw/ Varying Number of Trees')
    plt.xlabel('Number of Trees')
    plt.ylabel('Test Error Rate')
    plt.legend(['Random Forest', 'Decision Tree'])

    try:
        plt.savefig('Images/RF_vs_DT.png')

    except:
        plt.savefig('../Images/RF_vs_DT.png')

    plt.show()

    cf = metrics.confusion_matrix(ytest.flatten(), ypred)
    ax = plt.subplot()
    sns.heatmap(cf, 
                annot=True, 
                fmt='g', 
                cmap = 'Purples',
                cbar = False,                
                ax=ax);

    title = 'Confusion Matrix - Random Forest with ' + str(min_index) + ' Trees'
    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels');
    ax.set_title(title);
    ax.xaxis.set_ticklabels(['Not Returned', 'Returned']);
    ax.yaxis.set_ticklabels(['Not Returned', 'Returned'])

    try:
        plt.savefig('Images/CM_RF.png')

    except:
        plt.savefig('../Images/CM_RF.png')

    plt.show()

    print(f'*************** Random Forest ({min_index} trees) Summary ***************')
    print('Classes: ', rf_best.classes_)
    print('Number of Classes: ', rf_best.n_classes_)
    print('# of estimators: ', rf_best.n_estimators)
    print('# of features: ', rf_best.n_features_)
    c = calc_scores(cf, "Random Forest")
    print('--------------------------------------------------------')


    # Get feature importance using the mean decrease in impurity:
    # Feature importances are provided by the fitted attribute feature_importances_
    # and they are computed as the mean and standard deviation of accumulation 
    # of the impurity decrease within each tree.
    # std = np.std([tree.feature_importances_ for tree in rf_best.estimators_], axis=0)
    forest_importances = pd.DataFrame(zip(cols.tolist(), rf_best.feature_importances_.tolist()), 
                                      columns = ['feature', 'importance'])
    
    forest_importances_top10 = forest_importances.sort_values('importance', ascending = False).head(10)
    
    plt.figure(figsize = (8,8))

    plt.bar(forest_importances_top10['feature'],
             # Convert the accuracy values from a string to a float and remove
             # the '%' symbol at the end of it and sort the results in descending
             # order
             sorted(forest_importances_top10['importance'], reverse = True),
             color = 'Slateblue')
    
    # Set the y-axis
    plt.ylim([min(forest_importances_top10['importance']) - .05, 
              max(forest_importances_top10['importance']) + .05])
    # Tilt the x-labels
    plt.xticks(rotation = 45)
    plt.xlabel('Feature', fontsize = 14)
    plt.ylabel('Importance (Mean Accuracy Decrease)', fontsize = 14)
    plt.title('Feature Importance Using Permutation on Random Forest', fontsize = 18)
    
    plt.savefig('Images/Random_Forest_Feature_Importance.png', bbox_inches='tight')
    plt.show() 
    
    return c


# Adaboost
def classifier_adaboost(xtrain, xtest, ytrain, ytest):
    print(f"Running Adaboost...")
    adaboost_model = AdaBoostClassifier()
    
    # Run 10-fold cross-validation to see the average accuracy score
    cv = RepeatedStratifiedKFold(n_splits = 10,
                                 n_repeats = 3)
    n_scores = cross_val_score(adaboost_model, xtrain, ytrain,
                               scoring = 'precision',
                               cv = cv,
                               n_jobs = -1,
                               error_score = 'raise')
    
    print("Average Adaboost accuracy using 10-fold cross validation:", np.mean(n_scores))
    
    
    # define the grid of values to search
    print("Searching for optimal Adaboost parameters...")
    grid = dict()
    grid['n_estimators'] = [10, 50, 100, 500]
    grid['learning_rate'] = [0.0001, 0.001, 0.01, 0.1, 1.0]
    # define the grid search procedure
    grid_search = GridSearchCV(estimator = adaboost_model, 
                               param_grid = grid, 
                               n_jobs = -1, 
                               cv = cv, 
                               scoring = 'precision')
    # execute the grid search
    grid_result = grid_search.fit(xtrain, ytrain)
    
    # summarize the best score and configuration
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    optimal_adb = grid_search.best_estimator_
    ypred = optimal_adb.fit(xtrain, ytrain).predict(xtest)

    cf = metrics.confusion_matrix(ytest.flatten(), ypred)
    ax = plt.subplot()
    sns.heatmap(cf, 
                annot=True, 
                fmt='g', 
                cmap = 'Purples',
                cbar = False,                
                ax=ax);

    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix - Adaboost');
    ax.xaxis.set_ticklabels(['Not Returned', 'Returned']);
    ax.yaxis.set_ticklabels(['Not Returned', 'Returned'])

    try:
        plt.savefig('Images/CM_adaboost.png')

    except:
        plt.savefig('../Images/CM_adaboost.png')

    plt.show()

    c = calc_scores(cf, "Adaboost")

    return c 


xtrain, xtest, ytrain, ytest = data_prep(dogs_selected)

# Run all of our classifiers
clf_results = classifier_NB(xtrain, xtest, ytrain, ytest)
clf_results = pd.concat([classifier_KNN(xtrain, xtest, ytrain, ytest), clf_results])
clf_results = pd.concat([classifier_LR(xtrain, xtest, ytrain, ytest), clf_results])
clf_results = pd.concat([classifier_SVM(xtrain, xtest, ytrain, ytest), clf_results])
clf_results = pd.concat([classifier_KSVM(xtrain, xtest, ytrain, ytest), clf_results])
clf_results = pd.concat([classifier_NNet(xtrain, xtest, ytrain, ytest), clf_results])
clf_results = pd.concat([classifier_tree(xtrain, xtest, ytrain, ytest, cols), clf_results])
clf_results = pd.concat([classifier_RF(xtrain, xtest, ytrain, ytest), clf_results])
clf_results = pd.concat([classifier_adaboost(xtrain, xtest, ytrain, ytest), clf_results])


# try:
#     dfi.export(a, 'Images/ScoresResults.png')

# except:
#     dfi.export(a, '../Images/ScoresResults.png')

#%% Plot the results of our classifiers
np.random.seed(42)

def plot_classifier_accuracy(df, name = 'Classifiers_Final_Accuracy'):
    plt.figure(figsize = (8,8))

    plt.bar(clf_results['Classifier'],
             # Convert the accuracy values from a string to a float and remove
             # the '%' symbol at the end of it and sort the results in descending
             # order
             sorted(clf_results['Accuracy'].apply(lambda x: float(re.sub('%', '', x))),
                    reverse = True),
             color = 'Slateblue')
    
    # Set the y-axis
    plt.ylim([min(clf_results['Accuracy'].apply(lambda x: float(re.sub('%', '', x)))) - 5, 
              max(clf_results['Accuracy'].apply(lambda x: float(re.sub('%', '', x)))) + 5])
    # Tilt the x-labels
    plt.xticks(rotation = 45)
    plt.xlabel('Classifier', fontsize = 14)
    plt.ylabel('Accuracy (%)', fontsize = 14)
    plt.title('Accuracy Results of Various Classifiers', fontsize = 18)
    
    plt.savefig('Images/' + name + '.png', bbox_inches='tight')
    plt.show() 

plot_classifier_accuracy(clf_results)

#%%
# Now let's try to run the same but with only the columns we are more confident
# in from our requirements gathering sessions.
dogs_reduced = dogs_selected.drop(columns = ['MIX_BOOL',
                                         'multi_color',
                                         # 'num_colors',
                                         # 'contains_black',
                                         # 'contains_white',
                                         # 'contains_yellow',
                                         'new_this_week',
                                         'diarrhea',
                                         'uri',
                                         'tapeworm',
                                         'general_infection',
                                         'dog_park',
                                         'treated_vaccinated',
                                         'FT_FIXED',
                                         'car_sick',
                                         'hair_loss',
                                         'tapeworm',
                                         'ear_infection',
                                         'anaplasmosis',
                                         'ehrlichia',
                                         'spay_neuter'])

xtrain, xtest, ytrain, ytest = data_prep(dogs_reduced)

# Run all of our classifiers 
clf_reduced_results = classifier_NB(xtrain, xtest, ytrain, ytest)
clf_reduced_results = pd.concat([classifier_KNN(xtrain, xtest, ytrain, ytest), clf_reduced_results])
clf_reduced_results = pd.concat([classifier_LR(xtrain, xtest, ytrain, ytest), clf_reduced_results])
clf_reduced_results = pd.concat([classifier_SVM(xtrain, xtest, ytrain, ytest), clf_reduced_results])
clf_reduced_results = pd.concat([classifier_KSVM(xtrain, xtest, ytrain, ytest), clf_reduced_results])
clf_reduced_results = pd.concat([classifier_NNet(xtrain, xtest, ytrain, ytest), clf_reduced_results])
clf_reduced_results = pd.concat([classifier_tree(xtrain, xtest, ytrain, ytest, dogs_reduced.columns), clf_reduced_results])
clf_reduced_results = pd.concat([classifier_RF(xtrain, xtest, ytrain, ytest), clf_reduced_results])
clf_reduced_results = pd.concat([classifier_adaboost(xtrain, xtest, ytrain, ytest), clf_reduced_results])

plot_classifier_accuracy(clf_reduced_results, name = 'Classifiers_Final_Accuracy_Reduced_Variables')


#%% A Walkthrough example
# In this section we'll take two dogs, one returned and one not returned
# and see what the Random Forest model predicts for each
print("Prepping data...")
print("Imputing missing values...")
data = na_imputation(dogs_reduced)
y = data.iloc[:,-1]
X = data.iloc[:,0:-1]

# scale data
print("Scaling data...")
xdata = scale_arr(X)

# Bring these back together in dataframe form and merge the link back in
# dogs_fixed = pd.concat([pd.DataFrame(xdata, index = y.index, columns = cols), y], axis = 1) \
dogs_fixed = data \
                .merge(dogs_joined[['ID', 'DOG NAME', 'LINK']],
                       on = 'ID',
                       how = 'left')

# Get random dogs
ex_returned_dog = dogs_fixed[dogs_fixed['returned'] == 1].sample(1)
ex_notreturned_dog = dogs_fixed[dogs_fixed['returned'] == 0].sample(1)

# Get specific dogs in report
ex_returned_dog = dogs_fixed[dogs_fixed['ID'] == 'PICK-MD-14-0037']
ex_notreturned_dog = dogs_fixed[dogs_fixed['ID'] == 'MTHY-MD-15-0081']
ex_dogs = pd.concat([ex_returned_dog, ex_notreturned_dog], sort = False)

# See what Random Forest predicts for these two
# min_value = min(rf_misc)
# min_index = rf_misc.index(min_value)
# rf_best = RandomForestClassifier(n_estimators=min_index, random_state=24)
# rf_best.fit(xtrain, ytrain)
ex_pred = rf_best.predict(ex_dogs.drop(columns = ['ID', 'LINK', 'DOG NAME', 'returned']).to_numpy())

# Bring the predictions back in
ex_dogs = pd.merge(ex_dogs, pd.DataFrame(ex_pred, 
                                           columns = ['predicted_return'],
                                           index = ex_dogs.ID),
                   on = 'ID',
                   how = 'inner')

if ex_dogs.LINK.isna().sum() == 0 and all(ex_dogs.returned == ex_dogs.predicted_return):
    print("We good")
    ex_full = dogs_joined[dogs_joined['ID'].isin(ex_dogs['ID'])]
else:
    print("Keep running the code...")