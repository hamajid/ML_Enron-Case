#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

import numpy as np
import pandas as pd
import seaborn as sns


from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectPercentile
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from time import time
from sklearn.grid_search import GridSearchCV

import warnings

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','exercised_stock_options', 'total_stock_value', 'bonus',
                      'salary', 'deferred_income', 'long_term_incentive',
                      'restricted_stock', 'total_payments', 'shared_receipt_with_poi',
                      'loan_advances', 'expenses', 'from_poi_to_this_person', 'from_this_person_to_poi']

financial_features = ['salary', 'deferral_payments', 'total_payments',
                      'exercised_stock_options', 'bonus', 'restricted_stock',
                      'restricted_stock_deferred', 'total_stock_value', 'expenses',
                      'loan_advances', 'other', 'director_fees', 
                      'deferred_income', 'long_term_incentive']

email_features = ['to_messages', 'shared_receipt_with_poi', 'from_messages',
                  'from_this_person_to_poi', 'from_poi_to_this_person','email_address' ]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    enron_data = pickle.load(data_file)
df = pd.DataFrame.from_dict(enron_data, orient='index')
df = df.replace('NaN', np.nan)

# replacing NaN's by 0 for financial features :

df = df.replace('NaN', np.nan)
df[financial_features] = df[financial_features].fillna(0)

# replacing NaN's by 0 for email  features :

df[email_features] = df[email_features].fillna(df[email_features].median())

### Task 2: Remove outliers
### removing the total outlier
df.drop(axis=0, labels = 'TOTAL', inplace=True)


### Task 3: Create new feature(s)
df_newf = df.copy()
df_newf['to_poi_ratio'] = df.from_poi_to_this_person / df.to_messages
df_newf['from_poi_ratio'] = df.from_this_person_to_poi / df.from_messages

# Add new email features to the features list
features_list.append('to_poi_ratio')
features_list.append('from_poi_ratio')

### Store to my_dataset for easy export below.
my_dataset = df_newf.to_dict('index')

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
data = df.drop(['poi', 'email_address'], axis=1)
X = data.values
y = df.poi

# Generate the training and testing data
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=2)
skf.get_n_splits(X, y)

StratifiedKFold(n_splits=2, random_state=None, shuffle=False)
for train_index, test_index in skf.split(X, y):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


# Create training and testing set with 2 PCA components
pca = PCA(n_components=2)
pca.fit(X_train)
X_train_pca2 = pca.transform(X_train)
X_test_pca2 = pca.transform(X_test)
# Create training and testing set with 5 PCA components
pca = PCA(n_components=5)
pca.fit(X_train)
X_train_pca5 = pca.transform(X_train)
X_test_pca5 = pca.transform(X_test)


# Create training and testing set with top 30% of features
X_train_reduce30 = SelectPercentile(percentile=30).fit_transform(X_train, y_train)
X_test_reduce30 = SelectPercentile(percentile=30).fit_transform(X_test, y_test)
# Create training and testing set with top 10% of features
X_train_reduce10 = SelectPercentile().fit_transform(X_train, y_train)
X_test_reduce10 = SelectPercentile().fit_transform(X_test, y_test)

#Test accuracy of various classifiers
models = {"AdaBoost Classifier": AdaBoostClassifier(), "Gaussian Naive Bayes": GaussianNB(),
          "Decision Tree Classifier": tree.DecisionTreeClassifier(), "Random Forest Classifier": RandomForestClassifier()}

data = {"": X_train, "(30% of features)": X_train_reduce30, "(10% of features)": X_train_reduce10,
       "(PCA: 2 components)": X_train_pca2, "(PCA: 5 components)": X_train_pca5}

res = {}

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    for name, model in models.items():
        for scale, X in data.items():
            n = name + " " + scale
            clf = model
            accuracy = cross_val_score(clf, X_train, y_train).mean()

            y_pred = cross_val_predict(clf, X_train, y_train)

            rec = cross_val_score(clf, X_train, y_train, scoring=make_scorer(recall_score)).mean()
            prec = cross_val_score(clf, X_train, y_train, scoring=make_scorer(recall_score)).mean()
            f1 = cross_val_score(clf, X_train, y_train, scoring=make_scorer(recall_score)).mean()


            res[n] = {"MeanAccuracy": accuracy, "Precision": prec, "Recall": rec, "F1Score": f1}

results = pd.DataFrame.from_dict(res, orient="index")
results = results[["MeanAccuracy", "Precision", "Recall", "F1Score"]]

results


### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script.

#Search for best prameters
parameters = {
    'criterion': ('gini', 'entropy'),
    'min_samples_leaf':range(1, 50, 5),
    'max_depth': range(1, 10)
}

scorer = make_scorer(f1_score)
DTC = DecisionTreeClassifier(random_state=42)

clf = GridSearchCV(DTC, param_grid=parameters, scoring=scorer)
clf.fit(X_train_pca5, y_train)
clf.best_params_, clf.best_score_

## Run the Gaussian Naive Bayes algorithm since it has the best results

clf = DecisionTreeClassifier(splitter="random",
                             max_depth=4,
                             criterion='entropy',
                             min_samples_split=10,
                             max_leaf_nodes=6,
                             class_weight="balanced",
                             random_state=42)

clf.fit(X_train_pca5,y_train)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
