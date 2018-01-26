__author__ = 'amendrashrestha'

import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import numpy as np
import nltk
import re
import string
import traceback

import AliasBackend.utilities.IOProperties as props
import AliasBackend.utilities.IOReadWrite as IO


def classification():
    #
    # url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
    # names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    # dataframe = pd.read_csv(url, names=names)

    fv_dataframe = pd.read_csv(prop.feature_vector_sample_filepath)

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

    abs_result.to_csv(prop.feature_vector_with_abs_filepath)

    dataframe = pd.read_csv(prop.feature_vector_with_abs_filepath)



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

    vector = np.zeros((2, len(features)))

    for x in all_text:

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


            if col == len(features) - 1:
                col = 0
                break
            col += 1

    return vector

if __name__ == "__main__":
    # classification()
    # get_diff()
    fv_dataframe = create_feature_vector_temp("this is stest", "asfjdls lkajsdlk jalkfj klajsd l")
    df = pd.DataFrame(fv_dataframe)
    abs_fv = abs(df.diff()).dropna()
    print(abs_fv)