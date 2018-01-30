from flask import Flask, render_template, request, jsonify

from sklearn.externals import joblib
from sklearn.metrics.pairwise import cosine_similarity
import traceback
import pandas as pd
import numpy as np
import nltk

import string
import re
import warnings
warnings.filterwarnings("ignore")

import AliasBackend.utilities.IOProperties as props
import AliasBackend.utilities.IOReadWrite as IO

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


@app.route("/predict")
def predict():
    text1 = request.args.get('text1')
    text2 = request.args.get('text2')

    fv_dataframe = create_feature_vector(text1, text2)

    # array_length = fv_dataframe.shape[1]
    first_row = fv_dataframe[0][:-1]
    second_row = fv_dataframe[1][:-1]

    # print(first_row)
    # print(second_row)

    # cosine_sim = cosine_similarity(first_row, second_row)
    # print(cosine_sim)

    df = pd.DataFrame(fv_dataframe)
    abs_fv = abs(df.diff()).dropna()

    x_test = abs_fv.iloc[:,0:len(abs_fv.columns)-1]
    # y_test = abs_fv.iloc[:, -1]

    try:
        rf = joblib.load('static/model/cal_svm_finalized_model.sav')

        predicted_test_scores = rf.predict_proba(x_test)
        #accuracy = rf.score(x_test, y_test)
        # print(accuracy + "%")
        return jsonify(
            pred_class = rf.predict(x_test)[0],
            same_user_prob = str(predicted_test_scores[:, 0][0]),
            diff_user_prob = str(predicted_test_scores[:, 1][0]),
            cosine_sim = cosine_similarity(first_row, second_row)[0][0]
        )

    except ValueError:
        return jsonify(accuracy = "Please provide data!!")

def create_feature_vector(text1, text2):
    row = 0
    col = 0

    user = 1

    LIWC, word_lengths, digits, symbols, smilies, functions, user_id, features, header_feature = IO.FV_header()
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
            row += 1

        except Exception:
            traceback.print_exc()

    return vector

if __name__ == '__main__':
    app.run(debug=True)
