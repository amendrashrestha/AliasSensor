from flask import Flask, render_template, request

from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np
import nltk

import string
import jsonify
import re
import os

import AliasBackend.utilities.IOProperties as props
import AliasBackend.utilities.IOReadWrite as IO

static_folder_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


@app.route("/", methods=['POST'])
def predict():
    text1 = request.form.get('text1')
    text2 = request.form.get('text2')
    fv_dataframe = create_feature_vector(text1)

    abs_fv = pd.DataFrame(fv_dataframe)

    # abs_fv = abs(df.diff()).dropna()
    x_test = abs_fv.iloc[:,0:len(abs_fv.columns)-1]
    y_test = abs_fv.iloc[:, -1]

    print(x_test)
    print(y_test)

    try:
        rf = joblib.load('static/model/svm_finalized_model.sav')
        accuracy = rf.score(x_test, y_test)
        print("Accuracy: %.2f%%" % (accuracy * 100))

        # accuracy = evaluate(rf, x_test, y_test)
        return jsonify(accuracy)

    except ValueError:
        return jsonify("Please provide data!!")

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return jsonify(accuracy)

def create_feature_vector(x):
    row = 0
    col = 0

    user = 1

    LIWC, word_lengths, digits, symbols, smilies, functions, user_id, features, header_feature = IO.FV_header()


    # x = "länka till reportage ur massmedia. Det ska då vara sakliga reportage med integrations- och invandringspolitiska teman. @ 02:30-03:25. "
    vector = np.zeros((1, len(features)))
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


if __name__ == '__main__':
    app.run(debug=True)
