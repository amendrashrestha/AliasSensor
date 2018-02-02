__author__ = 'amendrashrestha'

import traceback
import re
import nltk
import numpy as np
from langdetect import detect

from sklearn.externals import joblib

import string

import AliasBackend.utilities.IOProperties as props
import AliasBackend.utilities.IOReadWrite as IO

def create_swedish_feature_vector(text1, text2):
    row = 0
    col = 0

    user = 1

    LIWC, word_lengths, digits, symbols, smilies, functions, user_id, features, header_feature = IO.FV_Swedish_header()
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

def detect_language(text):
    return detect(text)

def return_swe_result(x_test):
    rf = joblib.load('static/model/swe_cal_svm_finalized_model.sav')
    predicted_test_scores = rf.predict_proba(x_test)

    pred_class = float(rf.predict(x_test)[0])
    same_user_prob = str(predicted_test_scores[:, 0][0])
    diff_user_prob = str(predicted_test_scores[:, 1][0])

    return pred_class, same_user_prob, diff_user_prob

def return_eng_result(x_test):
    rf = joblib.load('static/model/eng_cal_rf_finalized_model.sav')
    predicted_test_scores = rf.predict_proba(x_test)

    pred_class = float(rf.predict(x_test)[0])
    same_user_prob = str(predicted_test_scores[:, 0][0])
    diff_user_prob = str(predicted_test_scores[:, 1][0])

    return pred_class, same_user_prob, diff_user_prob