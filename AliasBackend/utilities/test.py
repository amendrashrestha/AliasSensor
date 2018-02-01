__author__ = 'amendrashrestha'

import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import  CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.feature_selection import SelectPercentile, chi2

import numpy as np
import nltk
import re
import string
import traceback
import warnings
warnings.filterwarnings("ignore")

import AliasBackend.utilities.IOProperties as props
import AliasBackend.utilities.IOReadWrite as IO


def calibratedClassification():
    fv_dataframe = pd.read_csv(props.feature_vector_filepath)

    df = pd.DataFrame(fv_dataframe)

    abs_result_df = abs(df.diff()).dropna()
    X = abs_result_df.iloc[:,0:len(abs_result_df.columns)-1]
    # print(X)

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


def testing_model_with_SVM(x_train, x_test, y_train, y_test, kfold, percentile):

    penalty = 'l2'
    loss = 'squared_hinge'

    svc = LinearSVC(C=1.0, penalty=penalty, loss=loss, dual=True)

    # pipeline = Pipeline([
    #     ('featureselection', SelectPercentile(chi2, percentile=percentile)),
    #     ('model', svc)
    # ])

    # print(x_train)
    # print("--------")
    # print(y_train)

    # params = dict(model__C=[2 ** -12, 2 ** -9, 2 ** -7, 2 ** -5, 2 ** -3, 2 ** -1, 2 ** 1])
    # gs = GridSearchCV(estimator=pipeline, cv=kfold, param_grid=params)
    # print('\nTraining model with', str(percentile), "% features")
    clf = CalibratedClassifierCV(svc)
    clf.fit(x_train, y_train)

    # best_est = clf.best_estimator_
    #Saving model
    joblib.dump(clf, props.cal_svm_model_filename)

    print('Testing model')
    # Save the classifiers predictions and create confusion matrix

    accuracy = clf.score(x_test, y_test)
    # print(accuracy)
    print("Accuracy: %.2f%%" % (accuracy * 100))

def logistic_regression():
    fv_dataframe = pd.read_csv(props.feature_vector_filepath)

    df = pd.DataFrame(fv_dataframe)

    abs_result_df = abs(df.diff()).dropna()
    X = abs_result_df.iloc[:,0:len(abs_result_df.columns)-1]
    # print(X)

    Y = abs_result_df.iloc[:, -1]

    test_size = 0.2
    seed = 7
    kfold = 10

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)

    # Fit the model on 33%
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    # save the model to disk
    # filename = 'finalized_model.sav'
    # joblib.dump(model, filename)

    # some time later...

    # load the model from disk
    # loaded_model = joblib.load(prop.model_filename)
    result = model.score(X_test, Y_test)
    print(result)

def testing_cali_model():
    # 0 = same user
    # 1 = diff user
    text1 = "hej, skulle ominstallera datorn tänkte jag. Har gjort det tidigare men har nu glömt hur jag gjorde. Det jag vet är att jag inte behövde använda ngn skiva eller något annat. OM ngn kunde hjälpa hade det varit enormt bra,"
    text2 = "Knappt märkbart.Om du är så angelägen om säkerheten så ta en titt på inlägg #5581 gällande Comodo Bv. Värt att prova ?! Comodo har en sandbox inbyggd som går att använda på samma sätt som Sandboxie enligt tidigare länk. "

    text1 = "allokera allt allt som oftast"
    text2 = text1

    fv_dataframe = create_feature_vector_temp(text1, text2)
    # print(fv_dataframe[0])
    # print(fv_dataframe[1])
    df = pd.DataFrame(fv_dataframe)
    print(df)
    abs_fv = abs(df.diff()).dropna()
    # print(abs_fv)
    x_test = abs_fv.iloc[:,0:len(abs_fv.columns)-1]
    # y_test = abs_fv.iloc[:, -1]

    loaded_model = joblib.load(props.swedish_cal_svm_model_filename)

    # Predicted class labels from test features
    pred_class = loaded_model.predict(x_test)
    print(pred_class)
    if pred_class == 1:
        print("Label: " + "Diff User")
    else:
        print("Label: " + "Same User")

    # Predicted probabilities from test features
    predicted_test_scores = loaded_model.predict_proba(x_test)

    print("Classes" + str(loaded_model.classes_))
    # print(predicted_test_scores)
    print("Probability for label 0" + str(predicted_test_scores[:, 0]))
    print("Probability for label 1" + str(predicted_test_scores[:, 1]))

def classification():
    #
    # url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
    # names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    # dataframe = pd.read_csv(url, names=names)

    fv_dataframe = pd.read_csv(props.feature_vector_sample_filepath)

    df = pd.DataFrame(fv_dataframe)

    abs_result_df = abs(df.diff()).dropna()

    # abs_result.to_csv(prop.feature_vector_with_abs_filepath)

    # dataframe = pd.read_csv(prop.feature_vector_with_abs_filepath)

    # array = dataframe.values
    # print(len(dataframe.columns))
    X = abs_result_df.iloc[:,0:len(abs_result_df.columns)-1]
    # print(X)

    Y = abs_result_df.iloc[:, -1]
    # print(Y)

    test_size = 0.33
    seed = 7

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)

    print(Y_test)
    # Fit the model on 33%
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    # save the model to disk
    filename = 'finalized_model.sav'
    joblib.dump(model, filename)

    # some time later...

    # load the model from disk
    loaded_model = joblib.load(filename)
    result = loaded_model.score(X_test, Y_test)
    print(result)

def get_diff():
    data = np.array([
                [1,2,5,1],
                [3,4,9,1],
                [4,2,3,2],
                [2,1,5,2],
                [7,7,1,3],
                [3,2,8,3]

    ])
    header = ['a','b', 'c', 'd']

    # df = pd.read_csv(prop.feature_vector_filepath)

    df = pd.DataFrame(data,  columns = header)

    abs_result = abs(df.diff()).dropna()
    print(abs_result)
    print("-------------")

    abs_result.to_csv(props.feature_vector_with_abs_filepath)

    dataframe = pd.read_csv(props.feature_vector_with_abs_filepath)



    # for i in range(0, (len(df.index)-1)):
    #     first_row = df.iloc[i]
    #     second_row = df.iloc[i+1]
    #     diff = abs(first_row - second_row)
    #
    #     # print(diff)
    #     # df.to_csv(diff, ab)
    #     with open(prop.feature_vector_with_abs_filepath, 'a') as f:
    #         diff.to_csv(f, header=None)

        # with open(prop.feature_vector_with_abs_filepath, 'ab') as f_handle:
        #     df.to_csv(f_handle, diff)
        # print(diff)

def create_feature_vector_temp(text1, text2):
    row = 0
    col = 0

    user = 1

    LIWC, word_lengths, digits, symbols, smilies, functions, user_id, features, header_feature = IO.FV_header()
    all_text = []


    all_text.append(text1)
    all_text.append(text2)

    # x = "länka till reportage ur massmedia. Det ska då vara sakliga reportage med integrations- och invandringspolitiska teman. @ 02:30-03:25. "
    vector_all = []

    vector = np.zeros((len(all_text), len(features)))

    # print(vector)

    for x in all_text:
        # print(x)
        x = x.lower()
        split_text = x.split()
        text_size = len(split_text)

        tmp_x = str.maketrans({key: None for key in string.punctuation})
        x_wo_punct = x.translate(tmp_x)

        x_words = nltk.word_tokenize(x_wo_punct)
        # print(x_words)

        word_lengths_counts = nltk.FreqDist([len(tok) for tok in x_words])

        # for word, count in word_lengths_counts.most_common().sort():
        #     print(word, count)

        for feat in features:
            # print(feat)
            if col < len(LIWC):
                LIWC_filepath = props.LIWC_filepath + feat
                LIWC_words = IO.get_function_words(LIWC_filepath)
                count = 0
                try:
                    for single_word in LIWC_words:
                        count += sum(1 for i in re.finditer(single_word, x_wo_punct))
                        # print(feat, single_word, count)
                    avg_count = count / text_size
                    # print(avg_count)

                    vector[row][col] = avg_count
                except Exception:
                    traceback.print_exc()

            # Count word lengths
            elif col < len(LIWC) + len(word_lengths):
                if int(feat) in word_lengths_counts.keys():
                    vector[row][col] = word_lengths_counts.get(int(feat)) / text_size
                else:
                    vector[row][col] = 0

            # Count digits
            elif col < len(LIWC) + len(word_lengths) + len(digits):
                vector[row][col] = x_wo_punct.count(feat) / text_size

            # Count special symbols
            elif col < len(LIWC) + len(word_lengths) + len(digits) + len(symbols):
                vector[row][col] = x.count(feat) / text_size

            # Count smileys
            elif col < len(LIWC) + len(word_lengths) + len(digits) + len(symbols) + len(smilies):
                vector[row][col] = x.count(feat) / text_size
                # print(feat, x.count(feat))

            # Count functions words
            elif col < len(LIWC) + len(word_lengths) + len(digits) + len(symbols) + len(smilies) + len(functions):
                # vector[row][col] = len(re.findall(feat, " ".join(x).lower())) / text_size
                vector[row][col] = sum(1 for i in re.finditer(feat, x_wo_punct)) / text_size

            # # Adding userId

            elif col < len(LIWC) + len(word_lengths) + len(digits) + len(symbols) + len(smilies) + len(functions) + len(user_id):
                vector[row][col] = float(user)
                # print("User: " + str(user))

            if col == len(features) - 1:
                col = 0
                break
            col += 1
        row += 1
        # print(vector)
        # print("---------------------------------")

    return vector


def create_english_feature_vector(text1, text2):
    row = 0
    col = 0

    user = 1

    characters, word_lengths, digits, symbols, smilies, functions, pos_tags, user_id, features, header_feature = IO.FV_English_header()

    all_text = []
    all_text.append(text1)
    all_text.append(text2)

    # x = "länka till reportage ur massmedia. Det ska då vara sakliga reportage med integrations- och invandringspolitiska teman. @ 02:30-03:25. "

    vector = np.zeros((len(all_text), len(features)))

    for x in all_text:
        try:
            x = x.lower()
            split_text = x.split()
            text_size = len(split_text)

            tmp_x = str.maketrans({key: None for key in string.punctuation})
            x_wo_punct = x.translate(tmp_x)

            x_words = nltk.word_tokenize(x_wo_punct)
            # print(x_words)
            pos = nltk.FreqDist([b for (a, b) in nltk.pos_tag(x_wo_punct)])

            word_lengths_counts = nltk.FreqDist([len(tok) for tok in x_words])

            for feat in features:
                if col < len(characters):
                    vector[row][col] = " ".join(x).count(feat) / text_size

                # Count word lengths
                elif col < len(characters) + len(word_lengths):
                    if int(feat) in word_lengths_counts.keys():
                        vector[row][col] = word_lengths_counts.get(int(feat)) / text_size
                    else:
                        vector[row][col] = 0

                # Count digits
                elif col < len(characters) + len(word_lengths) + len(digits):
                    vector[row][col] = x_wo_punct.count(feat) / text_size

                # Count special symbols
                elif col < len(characters) + len(word_lengths) + len(digits) + len(symbols):
                    vector[row][col] = x.count(feat) / text_size

                # Count smileys
                elif col < len(characters) + len(word_lengths) + len(digits) + len(symbols) + len(smilies):
                    vector[row][col] = x.count(feat) / text_size
                    # print(feat, x.count(feat))

                # Count functions words
                elif col < len(characters) + len(word_lengths) + len(digits) + len(symbols) + len(smilies) + len(functions):
                    # vector[row][col] = len(re.findall(feat, " ".join(x).lower())) / text_size
                    vector[row][col] = sum(1 for i in re.finditer(feat, x_wo_punct)) / text_size

                # Count POS tags
                elif col < len(characters) + len(word_lengths) + len(digits) + len(symbols) + len(smilies) + len(functions)\
                        + len(pos_tags):
                    for tag in pos_tags:
                        if tag in pos.keys():
                            vector[row][col] = pos.get(tag) / text_size

                # # Adding userId
                elif col < len(characters) + len(word_lengths) + len(digits) + len(symbols) + len(smilies) + len(functions) +\
                        len(pos_tags) + len(user_id):
                    vector[row][col] = float(user)


                if col == len(features) - 1:
                    col = 0
                    break
                col += 1

            row += 1

        except Exception:
            traceback.print_exc()

    return vector

if __name__ == "__main__":
    # calibratedClassification()
    # testing_cali_model()
    # logistic_regression()
    # classification()
    # get_diff()
    text1 = "abaa de..."
    text2 = "abc dher"

    create_english_feature_vector(text1, text2)
    #
    # svm = joblib.load(props.svm_model_filename)
    # # clf = CalibratedClassifierCV(svm)
    # accuracy_predict = svm.predict(x_test)
    # print(accuracy_predict)
    #
    # accuracy = svm.score(x_test, y_test)
    # print(accuracy)
    # predicted_test_scores= clf.predict_proba(x_test)
    # print(predicted_test_scores)