__author__ = 'amendrashrestha'

import os
import time
import traceback

import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

import AliasBackend.model.dbScript as db
import AliasBackend.utilities.IOProperties as prop
from AliasBackend.main.featureCreator import StyloFeatures
from AliasBackend.utilities.test_evaluator import evaluator


def init():
    users = db.get_users()
    # print(len(users)) -- 4713
    user_id = 1
    print("Creating Stylometric features ..... \n")

    for single_user in tqdm(users):
        # print(single_user)
        # single_user = "darknesss"
        posts = db.get_user_post(single_user)

        StyloFeatures(user_id, posts)
        user_id += 1

        # print(len(User_A), len(User_B))

def classification():
    fv_dataframe = pd.read_csv(prop.feature_vector_filepath)

    try:

        df = pd.DataFrame(fv_dataframe)

        abs_result_df = abs(df.diff()).dropna()
        # print(len(abs_result_df.columns))

        # train, test = split_dataset(abs_result_df, 0.7)
        #
        # x_train = train.iloc[:,0:len(abs_result_df.columns)-1]
        # y_train = train.iloc[:, -1]
        #
        # x_test = test.iloc[:,0:len(abs_result_df.columns)-1]
        # y_test = test.iloc[:, -1]

        X = abs_result_df.iloc[:,0:len(abs_result_df.columns)-1]
        Y = abs_result_df.iloc[:, -1]

        test_size = 0.2
        seed = 7
        kfold = 10

        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
        # X_train, Y_train = split_dataset(abs_result_df, 0.7)

        # testing_model_with_RandFor(x_train, y_train, x_test, y_test)

        # print("Preprocessing finished!")
        # percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        # #
        percentiles = [100]
        for p in percentiles:
            testing_model_with_SVM(X_train, X_test, Y_train, Y_test, kfold, p)

    except Exception:
        traceback.print_exc()

    # Fit the model on 33%
    # model = LogisticRegression()
    # model.fit(X_train, Y_train)
    # # save the model to disk
    # filename = 'finalized_model.sav'
    # joblib.dump(model, filename)
    #
    # # some time later...
    #
    # # load the model from disk
    # loaded_model = joblib.load(prop.model_filename)
    # result = loaded_model.score(X_test, Y_test)
    # print(result)


def testing_model_with_SVM(x_train, x_test, y_train, y_test, kfold, percentile):
    start_train = time.time()
    penalty = 'l2'
    loss = 'squared_hinge'

    svc = LinearSVC(C=1.0, penalty=penalty, loss=loss, dual=True)
    if kfold == 1:
        # kfold = ShuffleSplit(test_size=0.20, n_iter=1, random_state=0)
        kfold = ShuffleSplit(test_size=0.20, random_state=0)

    pipeline = Pipeline([
        ('featureselection', SelectPercentile(chi2, percentile=percentile)),
        ('model', svc)
    ])

    # print(x_train)
    # print("--------")
    # print(y_train)

    params = dict(model__C=[2 ** -12, 2 ** -9, 2 ** -7, 2 ** -5, 2 ** -3, 2 ** -1, 2 ** 1])
    gs = GridSearchCV(estimator=pipeline, cv=kfold, param_grid=params)
    # print('\nTraining model with', str(percentile), "% features")
    gs.fit(x_train, y_train)

    best_est = gs.best_estimator_
    #Saving model
    joblib.dump(best_est, prop.svm_model_filename)

    print('Time training:', time.time() - start_train)

    # logger_params = str(percentile) + '_' + str(kfold) + '_' + str(min_df) + '_' + str(max_df)
    # logger_params = str(percentile) + '_' + str(kfold)
    # svm_training_<n_features>_<kfold>_<min_df>_<max_df>.log
    # training_logger_name = os.environ['HOME'] + '/Desktop/AliasMatching/log/svm_training_' + logger_params + '.log'
    # svm_testing_<n_features>_<kfold>_<min_df>_<max_df>.log
    testing_logger_name = os.environ['HOME'] + '/Desktop/AliasMatching/log/svm_testing_' + "logger_params" + '.log'


    # with open(training_logger_name, 'w') as f:
    #     f.write('Result from grid search cross validation:\n')
    #     for k in gs.cv_results_.keys():
    #         f.write(k + "\t" + str(gs.cv_results_[k]) + '\n')
    #     f.write('\nBest estimator parameters:\n')
    #     f.write(str(best_est.get_params()))

    print('Testing model')
    # Save the classifiers predictions and create confusion matrix
    y_test_pred = best_est.predict(x_test)
    duration = time.time() - start_train
    evaluator(testing_logger_name, [1, 2], [1, 2], y_test, y_test_pred, duration)

    accuracy = best_est.score(x_test, y_test)
    print("Accuracy: %.2f%%" % (accuracy * 100))

def testing_model_with_RandFor(x_train, y_train, x_test, y_test):
    start_train = time()
    # build a classifier
    clf = RandomForestClassifier()

    # print(x_train)
    # print("--------")
    # print(y_train)
    # get some data
    # digits = load_digits()
    # x_train, y_train = digits.data, digits.target
    # print(x_train)
    # print("--------")
    # print(y_train)

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # use a full grid over all parameters
    param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               "criterion": ["gini", "entropy"]}

    gs = GridSearchCV(estimator = clf, param_grid=param_grid, cv = 3, n_jobs = -1, verbose = 2)
    gs.fit(x_train, y_train)

    best_par = gs.best_params_
    best_grid = gs.best_estimator_

    grid_accuracy = evaluate(best_grid, x_test, y_test)
    print(grid_accuracy)

    # best_est = gs.best_estimator_
    # #Saving model
    # joblib.dump(best_est, prop.rf_model_filename, compress = 1)
    #
    # print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
    #   % (time() - start_train, len(gs.cv_results_['params'])))
    # report(gs.cv_results_)
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def split_dataset(dataset, train_percentage):
    # Split dataset into train and test dataset
    train, test = train_test_split(dataset, train_size=train_percentage)
    return train, test