#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

'''
Project: Investigate Fraud in Enron Emails

Richard Smith
March 2021

poi.id.py

See readme.txt for description of Python version and package environment used,
  as well as rationale for the decision.

Contents: (use "find" to navigate)
Pre-processing
  Task 1: Select what features you'll use
    1-A - features_list
  Task 2: Remove outliers 
    2-A - outlier removal
  Task 2.5: Data cleaning, checking, cleaning
    2-B - data cleaning
    2-C - data checking
    2-D - cleaning problem rows
  Task 3: Create new feature(s)
    3-A - feature creation
Classifier testing
  Task 4: Try a variety of classifiers
    4-A - Function definition for classifier testing, evaluation, validation
    4-B - Using iteration over a list of classifiers with Pipeline
    4-B - Checking feature importances
      4-B1 - Extracting features and labels from dataset for local testing
      4-B2 - by DecisionTree feature_importances_ ('Gini' impurity method)
      4-B3 - by SelectKBest scores_ ('ANOVA' f-values)
  Task 5: Tune your classifier to achieve better than .3 precision and recall
    5-A - Using GridSearchCV with Pipelined SelectKBest, DecisionTreeClassifier
    5-B - Using GridSearchCV with only DecisionTreeClassifier
    5-C - Checking feature_importances_ for tuned DecisionTreeClassifier
    5-D - Testing tuned parameters in classifiers
      5-D1 - with pipelined SelectKBest, DecisionTreeClassifier
      5-D2 - with only DecisionTreeClassifier, manual feature selection
      5-D3  -with only DecisionTreeClassifier, modified tuned parameter values,
               and manual feature selection
Output production
  Task 6: Dump your classifier, dataset, and features list
    6-A - Dumping "my_classifier.pkl", "my_dataset.pkl", "my_feature_list.pkl"
            via tester.dump_classifier_and_data

'''
# console output title splash
print("                        __          __      __")
print("                       /_/         /_/     / /")
print("      ______  ______  __          __  ____/ /     ______  __  _ ")
print("     / __  / / __  / / /         / / / __  /     / __  / / / / /")
print("    / /_/ / / /_/ / / / ______  / / / /_/ / __  / /_/ / / (_/ /")
print("   / ____/ /_____/ /_/ /_____/ /_/ /_____/ /_/ / ____/  L__  /")
print("  / /                                         / /       __/ /")
print(" /_/                                         /_/       /___/\n")
print("Edited to meet project requirements for:")
print(" 'Investigate Fraud in Enron Emails'\n\n")
###############################################################################
###   Task 1: Select what features you'll use.
###   features_list is a list of strings, each of which is a feature name.
###   The first feature must be "poi".
###############################################################################
# 1-A: Including all features except email_address
# note: this list is only used in section 0-E
#         features will be selected for classifiers via SelectKBest, later
features_list = ['poi',
                 'salary',
                 'bonus',
                 'long_term_incentive',
                 'expenses',
                 'director_fees',
                 'other',
                 'loan_advances',
                 'deferred_income',
                 'deferral_payments',
                 'total_payments',
                 'restricted_stock_deferred',
                 'exercised_stock_options',
                 'restricted_stock',
                 'total_stock_value',
                 'from_messages',
                 'to_messages',
                 'from_poi_to_this_person',
                 'from_this_person_to_poi',
                 'shared_receipt_with_poi',
                 'from_poi_to_messages_ratio',
                 'to_poi_from_messages_ratio',
                 'shared_receipt_to_messages_ratio']
set()
### Load the dictionary containing the dataset
print("Loading data... ", end='')
with open("final_project_dataset.pkl", "rb") as data_file:
  data_dict = pickle.load(data_file)
print("done.\n")

print(" ---------------------------------------------------")
print("--- Data cleaning, checking, and feature creation ---")
print(" ---------------------------------------------------\n")

###############################################################################
### Task 2: Remove outliers 
###############################################################################
# 2-A
print("Removing bad rows... ", end='')
# removing aggregate row
data_dict.pop('TOTAL', 0)
# removing non-person row
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)

# May or may not remove this row, depending on classifier testing results
## removing almost-all-zeroes row
# data_dict.pop('LOCKHART EUGENE E')
print("done.")

###############################################################################
### Task 2.5: Data cleaning, checking, cleaning
###############################################################################
# 2-B
# removing email addresses prior to data check (unused feature)
print("Removing email addresses... ", end='')
for k in data_dict.keys():
  if 'email_address' in data_dict[k].keys():
    data_dict[k].pop('email_address')
print("done.")

# 2-C - Data Checking
print("Checking data for issues related to 'total_payments'... ", end='')
payment_financial_features = ['salary',
                              'bonus',
                              'long_term_incentive',
                              'expenses',
                              'director_fees',
                              'other',
                              'loan_advances',
                              'deferred_income',
                              'deferral_payments']
problem_entries = {}
# Iterate over each row, check sum of above features against total_payments,
#   rows with mismatch added to problem_entries
for k in data_dict.keys():
  total_payments_check = 0
  for d in data_dict[k]:
    if d in payment_financial_features and data_dict[k][d] != 'NaN':
      total_payments_check += data_dict[k][d]
  if data_dict[k]['total_payments'] != 'NaN' and \
                        total_payments_check != data_dict[k]['total_payments']:
    problem_entries[k] = data_dict[k]

from pprint import pprint as pp
if len(problem_entries):
  print("found!")
  print("  Rows with issues related to 'total_payments' found:")
  pp(problem_entries)
else:
  print("none.")
print('')

# Due to problems found with two entries, 'BELFER ROBERT' and
#   'BHATNAGAR SANJAY', manually referencing /tools/enron61702insiderpay.pdf to
#   check data by eye.

# For 'BELFER ROBERT', lines marked with '#' were affected by an apparent shift
#   in values, corrected here with values from reference.
# Email data left as-is.
belfer_corrected = {'bonus': 'NaN',
                    'deferral_payments': 0,                   #
                    'deferred_income': -102500,               #
                    'director_fees': 102500,                  #
                    'exercised_stock_options': 0,             #
                    'expenses': 3285,                         #
                    'from_messages': 'NaN',
                    'from_poi_to_this_person': 'NaN',
                    'from_this_person_to_poi': 'NaN',
                    'loan_advances': 'NaN',
                    'long_term_incentive': 'NaN',
                    'other': 'NaN',
                    'poi': False,
                    'restricted_stock': 44093,                #
                    'restricted_stock_deferred': -44093,      #
                    'salary': 'NaN',
                    'shared_receipt_with_poi': 'NaN',
                    'to_messages': 'NaN',
                    'total_payments': 3285,                   #
                    'total_stock_value': 0}                   #

# Likewise, for 'BHATNAGAR SANJAY', lines marked with '#' were affected by an
#   apparent shift in data, corrected here with values from reference.
# Email data left as-is.
bhatnagar_corrected = {'bonus': 'NaN',
                       'deferral_payments': 'NaN',
                       'deferred_income': 'NaN',
                       'director_fees': 0,                    #
                       'exercised_stock_options': 15456290,   #
                       'expenses': 137864,                    #
                       'from_messages': 29,
                       'from_poi_to_this_person': 0,
                       'from_this_person_to_poi': 1,
                       'loan_advances': 'NaN',
                       'long_term_incentive': 'NaN',
                       'other': 0,                            #
                       'poi': False,
                       'restricted_stock': 2604490,           #
                       'restricted_stock_deferred': -2604490, #
                       'salary': 'NaN',
                       'shared_receipt_with_poi': 463,
                       'to_messages': 523,
                       'total_payments': 137864,              #
                       'total_stock_value': 15456290}         #

# Assigning corrected rows to dataset
print('Updating data with corrections... ', end='')
data_dict['BELFER ROBERT'] = belfer_corrected
data_dict['BHATNAGAR SANJAY'] = bhatnagar_corrected
print("done.")

# Repeating check to verify changes
print("Re-checking data for issues related to 'total_payments'... ", end='')
problem_entries = {}
for k in data_dict.keys():
  total_payments_check = 0
  for d in data_dict[k]:
    if d in payment_financial_features and data_dict[k][d] != 'NaN':
      total_payments_check += data_dict[k][d]
  if data_dict[k]['total_payments'] != 'NaN' and \
                        total_payments_check != data_dict[k]['total_payments']:
    problem_entries[k] = data_dict[k]

if len(problem_entries):
  print("found!")
  print("  Rows with issues related to 'total_payments' found:")
  pp(problem_entries)
else:
  print("none.")

###############################################################################
### Task 3: Create new feature(s) 
###############################################################################
# 3-A Feature Creation

# Presence of 'NaN' for any component features results in 'NaN' values for
#   created features which require them.
print("Creating features... ", end='')
for k in data_dict.keys():
  from_messages = True if \
    (data_dict[k]['from_messages'] != 'NaN') else False
  to_messages = True if \
    (data_dict[k]['to_messages'] != 'NaN') else False
  to_poi = True if \
    (data_dict[k]['from_this_person_to_poi'] != 'NaN') else  False
  from_poi = True if \
    (data_dict[k]['from_poi_to_this_person'] != 'NaN') else False
  shared_receipt = True if \
    (data_dict[k]['shared_receipt_with_poi'] != 'NaN') else False

  # ratio of emails sent to PoIs to emails sent generally:
  # to_poi_from_messages_ratio = from_this_person_to_poi / from_messages
  if to_poi and from_messages:
    data_dict[k]['to_poi_from_messages_ratio'] = \
       data_dict[k]['from_this_person_to_poi'] / data_dict[k]['from_messages']
  else:
    data_dict[k]['to_poi_from_messages_ratio'] = 'NaN'

  # ratio of emails received from PoIs to emails received generally:
  # from_poi_to_messages_ratio = from_poi_to_this_person / to_messages
  if from_poi and to_messages:
    data_dict[k]['from_poi_to_messages_ratio'] = \
          data_dict[k]['from_poi_to_this_person'] / data_dict[k]['to_messages']
  else:
    data_dict[k]['from_poi_to_messages_ratio'] = 'NaN'
  
  # ratio of emails having shared recipt with PoIs to emails received generally:
  # shared_receipt_to_messages_ratio = shared_receipt_with_poi / to_messages
  if shared_receipt and to_messages:
    data_dict[k]['shared_receipt_to_messages_ratio'] = \
       data_dict[k]['shared_receipt_with_poi'] / data_dict[k]['to_messages']
  else:
    data_dict[k]['shared_receipt_to_messages_ratio'] = 'NaN'
print("done.\n")

###############################################################################
### Task 4: Try a variety of classifiers 
###############################################################################
from sklearn.neighbors         import KNeighborsClassifier
from sklearn.tree              import DecisionTreeClassifier
from sklearn.naive_bayes       import GaussianNB
from sklearn.ensemble          import AdaBoostClassifier
from sklearn.model_selection   import StratifiedShuffleSplit

# 4-A Function definition for classifier testing, validation, evaluation
def classifier_test(clf, dataset, feature_list, folds = 1000):
  '''
  Based on code used in tester.py, with
equivalent functionality, this function
evaluates classifier performance through
cross-validation via
StratifiedShuffleSplit(), default 1000
splits for training and testing sets.
  Written primarily for personal
comprehension of the testing method used
in grading results, and to apply the
same metrics used in grading to
validation and evaluation of classifiers.

parameters:

clf:
  sklearn classifier, must support *.fit,
    *.predict
  
dataset:
  object compatible with Python dict,
    must have key entries containing
    features and values compatible with
    feature_list, will be processed by.
    feature_format.featureFormat()

feature_list:
  Python list, must contain strings
    matching features present in dict
    passed to 'dataset'.
  
folds:
  integer, default 1000, controls splits
    applied for cross validation via
    StratifiedShuffleSplit

output:
  Displays predictions made and
    performance results:
    Accuracy, Precision, Recall, F1, F2
  '''
  data = featureFormat(dataset, feature_list, sort_keys = True)
  labels, features = targetFeatureSplit(data)
  cv = StratifiedShuffleSplit(n_splits=folds, random_state = 42)
  true_neg  = 0
  false_neg = 0
  true_pos  = 0
  false_pos = 0
  for train_idx, test_idx in cv.split(features, labels):
    features_train = []
    labels_train   = []
    features_test  = []
    labels_test    = []
    for ii in train_idx:
      features_train.append(features[ii])
      labels_train.append(labels[ii])
    for jj in test_idx:
      features_test.append(features[jj])
      labels_test.append(labels[jj])

    # fit the classifier using training set, and test on test set
    clf.fit(features_train, labels_train)
    predictions = clf.predict(features_test)
    for prediction, truth in zip(predictions, labels_test):
      if prediction == 0 and truth == 0:
        true_neg += 1
      elif prediction == 0 and truth == 1:
        false_neg += 1
      elif prediction == 1 and truth == 0:
        false_pos += 1
      elif prediction == 1 and truth == 1:
        true_pos += 1
      else:
        print("Warning: Found a predicted label not == 0 or 1.")
        print("All predictions should take value 0 or 1.")
        print("Evaluating performance for processed predictions:")
        break
  try:
    total_pred = true_neg + false_neg + false_pos + true_pos
    accuracy = 1.0 * (true_pos + true_neg) / total_pred
    precision = 1.0 * true_pos / (true_pos + false_pos)
    recall = 1.0 * true_pos / (true_pos + false_neg)
    f1 = 2.0 * true_pos / (2 * true_pos + false_pos + false_neg)
    f2 = (1 + 2.0 * 2.0) * precision * recall / (4 * precision + recall)
    print(clf)
    print("  Predictions: %d" % total_pred)
    print("  Accuracy: %.5f\n  Precision: %.5f  Recall: %.5f" % \
          (accuracy, precision, recall))
    print("  F1: %.5f  F2: %.5f" % (f1, f2), "\n")
  except:
    print("Performance calculations failed.")
    print("Precision or recall may be undefined (no true positives).")


print(" ------------------------")
print("--- Classifier testing ---")
print(" ------------------------\n")

# 4-B  Iteration over a list of classifiers
# (see references.txt for code example source)
classifiers = [KNeighborsClassifier(),
               DecisionTreeClassifier(),
               GaussianNB()]

print("Trying several classifiers with default settings for comparison...\n")
for classifier in classifiers:
  classifier_test(classifier, data_dict, features_list)

###############################################################################
### Checking feature importances
###############################################################################
# 4-B
print(" ---------------------------------")
print("--- Checking feature importance ---")
print(" ---------------------------------\n")

# 4-B1: Extracting features and labels from dataset for local testing
print("Extracting features and labels... ", end='')
data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
print("done.\n")

# 4-B2: by DecisionTreeClassifier feature_importances_

print("DecisionTreeClassifier feature importances")
print("  ('Gini' impurity, only scores > 0.0)")

clf = DecisionTreeClassifier().fit(features, labels)

# sorting features_list copy by DecisionTreeClassifier.feature_importances_
# (see references.txt for code example used)
DTC_scores = sorted(zip(list(clf.feature_importances_),
                        features_list[1:]))[::-1]
for i in range(len(DTC_scores)):
  if DTC_scores[i][0] > 0:
    print(" ", i+1, "- '%s'" % DTC_scores[i][1],
          "\n          %.5f"   % DTC_scores[i][0])

# 4-B2: by SelectKBest scores_ ('ANOVA' f-values)

from sklearn.feature_selection import SelectKBest, f_classif

print("\nSelectKBest feature importances"
print("  (ANOVA f-values, all)")

skb = SelectKBest(f_classif).fit(features, labels)

# sorting features_list copy by SelectKBest.scores_
# (see references.txt for code example used)
skb_scores = sorted(zip(list(skb.scores_), features_list[1:]))[::-1]
for i in range(len(skb_scores)):
  print(" ", i+1, "- '%s'" % skb_scores[i][1],
        "\n          %.5f"   % skb_scores[i][0])

print('')

###############################################################################
### Task 5: Tune your classifier to achieve better than .3 precision and recall
###############################################################################
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline        import Pipeline

print(" --------------------------------------")
print("--- Optimizing classifier parameters ---")
print(" --------------------------------------\n")

# 5-A Using GridSearchCV with pipelined SelectKBest, DecisionTreeClassifier

tune_pipe1 = Pipeline([('feature_selection', SelectKBest(f_classif)),
                       ('clf', DecisionTreeClassifier())])
grid_params1 = {'feature_selection__k' : (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                         13, 14, 15, 16, 17, 18, 19, 20),
                'clf__criterion' : ('gini', 'entropy'),
                'clf__min_samples_split' : (2, 3, 4, 5)}

print("Trying GridSearchCV with")
pp(tune_pipe1)
print("and parameters")
pp(grid_params1)

tuner1 = GridSearchCV(tune_pipe1, grid_params1, scoring = 'f1', cv = 10,
                      n_jobs = -1)
tuner1.fit(features, labels)

print("Best parameters for maximizing 'f1':")
pp(tuner1.best_params_)

print("\n--------------------------------------------------------------------")

# 5-B Using GridSearchCV with only DecisionTreeClassifier

clf =  DecisionTreeClassifier()

grid_params2 = {'criterion' : ('gini', 'entropy'),
                'min_samples_split' : (2, 3, 4, 5)}

print("\nTrying GridSearchCV with", clf)
print("over parameters:")
pp(grid_params2)

tuner2 = GridSearchCV(clf, grid_params2, scoring = 'f1', cv = 10, n_jobs = -1)
tuner2.fit(features, labels)

print("Best parameters for maximizing 'f1':")
pp(tuner2.best_params_)

print("\n--------------------------------------------------------------------")

# 5-C Checking feature_importances_ for tuned DecisionTreeClassifier

print("\nTuned DecisionTreeClassifier feature importances")
print("('%s'-based, scores < 0.06 omitted)" % tuner2.best_params_['criterion'])
# sorting features_list copy by DecisionTreeClassifier.feature_importances_
# (see references.txt for code example used)
DTC_scores = sorted(zip(list(tuner2.best_estimator_.feature_importances_),
                               features_list[1:]))[::-1]
tuned_features = ['poi']
for i in range(len(DTC_scores)):
  if DTC_scores[i][0] > 0.06:
    best_features.append(DTC_scores[i][1])
    print(" ", i+1, "- '%s'" % DTC_scores[i][1],
          "\n          %.5f"   % DTC_scores[i][0])

print('Tuned features:')
pp(tuned_features[1:])
print('')

###############################################################################
### Testing tuned parameters in pipelined classifier
###############################################################################
print(" -----------------------------------------------")
print("--- Testing classifiers with tuned parameters ---")
print(" -----------------------------------------------\n")
# 5-D Testing tuned parameters in pipelined classifiers

# 5-D1 with pipelined SelectKBest, DecisionTreeClassifier

# passing optimized parameter values to respective steps
best_k = tuner1.best_params_['feature_selection__k']
best_criterion = tuner1.best_params_['clf__criterion']
best_mss = tuner1.best_params_['clf__min_samples_split']
skb = SelectKBest(k = best_k)
tuned_DTC_clf = DecisionTreeClassifier(criterion = best_criterion,
                                       min_samples_split = best_mss)
# passing steps to pipeline
pipe = Pipeline(steps = [('skb', skb),
                         ('clf', tuned_DTC_clf)])

print("Trying SelectKBest, DecisionTreeClassifier with tuned parameters...")
classifier_test(pipe, data_dict, features_list)

print("--------------------------------------------------------------------\n")

# 5-D2 with only DecisionTreeClassifier and automated feature selection
# (selection based on tuned DecisionTreeClassifier.feature_importances_)

# passing optimized parameter values to DecisionTreeClassifier
best_criterion = tuner2.best_params_['criterion']
best_min_samples_split = tuner2.best_params_['min_samples_split']
clf = DecisionTreeClassifier(criterion = best_criterion,
                             min_samples_split = best_min_samples_split)

print("Trying manual feature selection and DecisionTreeClassifier with tuned")
print("  parameters and automated feature selection")
print("  (GridSearchCV.best_estimator_.feature_importances_ > 0.06")
print("Features used:")
pp(best_features[1:])
classifier_test(clf, data_dict, tuned_features)

print("--------------------------------------------------------------------\n")

# 5-D3 with only DecisionTreeClassifier, modified tuned parameter values,
# and manual feature selection
# (selection based on untuned DecisionTreeClassifier.feature_importances_)

manual_features = ['poi',
                   'other',
                   'from_poi_to_this_person',
                   'expenses',
                   'bonus',
                   'to_poi_from_messages_ratio',
                   'shared_receipt_with_poi']


# passing semi-optimized parameter values (chosen according to but not dictated
# by GridSearchCV results) to DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion = 'entropy',
                             min_samples_split = 2)

print("Trying manual feature selection and DecisionTreeClassifier with tuned")
print("  parameters and manual feature selection")
print("  (DecisionTreeClassifier.feature_importances_ > 0.06")
print("Features used:")
pp(manual_features[1:])
classifier_test(clf, data_dict, manual_features)
###############################################################################
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
###############################################################################
# 6-A
import tester
tester.dump_classifier_and_data(clf, data_dict, manual_features)
tester.main()
