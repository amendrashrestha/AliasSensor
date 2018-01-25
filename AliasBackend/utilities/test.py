__author__ = 'amendrashrestha'

import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import numpy as np

import utilities.IOProperties as prop

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



if __name__ == "__main__":
    classification()
    # get_diff()