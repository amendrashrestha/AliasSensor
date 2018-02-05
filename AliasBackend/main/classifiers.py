__author__ = 'amendrashrestha'

import traceback

import pandas as pd
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn import model_selection
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn import preprocessing

import AliasBackend.utilities.IOProperties as props

def calibratedClassification():
    try:
        fv_dataframe = pd.read_csv(props.englsih_feature_vector_filepath)

        df = pd.DataFrame(fv_dataframe)

        abs_result_df = abs(df.diff()).dropna()
        X = abs_result_df.iloc[:,0:len(abs_result_df.columns)-1]

        Y = abs_result_df.iloc[:, -1]

        lab_enc = preprocessing.LabelEncoder()
        Y = lab_enc.fit_transform(Y)

        test_size = 0.2
        seed = 7
        kfold = 10

        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)

        testing_model_with_RandFor(X_train, X_test, Y_train, Y_test, kfold)

    except Exception:
        traceback.print_exc()

def testing_model_with_SVM(x_train, x_test, y_train, y_test):

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

def testing_model_with_RandFor(x_train, x_test, y_train, y_test):

    rfc = RandomForestClassifier(n_estimators = 300, n_jobs=-1, max_features= 'sqrt' , oob_score = True)

    calibrated_rfc = CalibratedClassifierCV(rfc, method='isotonic')

    calibrated_rfc.fit(x_train, y_train)

    joblib.dump(calibrated_rfc, props.swedish_cal_rf_model_filename)

    predictions = calibrated_rfc.predict(x_test)

    print("Test Accuracy  :: ", accuracy_score(y_test, predictions))
    print(" Confusion matrix ", confusion_matrix(y_test, predictions))