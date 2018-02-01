__author__ = 'amendrashrestha'

import traceback

import pandas as pd
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn import model_selection
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectPercentile, chi2

import AliasBackend.utilities.IOProperties as props
import AliasBackend.utilities.test_evaluator as eval

def calibratedClassification():
    try:
        fv_dataframe = pd.read_csv(props.englsih_feature_vector_filepath)

        df = pd.DataFrame(fv_dataframe)

        abs_result_df = abs(df.diff()).dropna()
        X = abs_result_df.iloc[:,0:len(abs_result_df.columns)-1]

        Y = abs_result_df.iloc[:, -1]

        test_size = 0.2
        seed = 7
        kfold = 10

        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)


        testing_model_with_SVM(X_train, X_test, Y_train, Y_test, kfold)

    except Exception:
        traceback.print_exc()

def testing_model_with_SVM(x_train, x_test, y_train, y_test, kfold):

    penalty = 'l2'
    loss = 'squared_hinge'

    try:
        svc = LinearSVC(C=1.0, penalty=penalty, loss=loss, dual=True)
        clf = CalibratedClassifierCV(svc, method="sigmoid")

        clf.fit(x_train, y_train)

        #Saving model
        joblib.dump(clf, props.english_cal_svm_model_filename)

        print('Testing model')
        # Save the classifiers predictions and create confusion matrix
        y_test_pred = clf.predict(x_test)

        accuracy = clf.score(x_test, y_test)
        # print(accuracy)
        print("Accuracy: %.2f%%" % (accuracy * 100))

    except Exception:
        traceback.print_exc()