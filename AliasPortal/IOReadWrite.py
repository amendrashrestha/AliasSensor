__author__ = 'amendrashrestha'

import traceback
import re
import nltk
import numpy as np
import os

from sklearn.externals import joblib

import string

import AliasPortal.IOProperties as props

def create_swedish_feature_vector(text1, text2):
    row = 0
    col = 0

    user = 1

    LIWC, characters, word_lengths, digits, symbols, smilies, functions, user_id, features, header_feature = FV_Swedish_header()
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

            text_length = len(x)

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
                    LIWC_words = get_function_words(LIWC_filepath)
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

                # Count swedish alphabates
                elif col < len(LIWC) + len(characters):
                    try:
                        vector[row][col] = " ".join(x).count(feat) / text_length

                    except Exception:
                        traceback.print_exc()

                # Count word lengths
                elif col < len(LIWC) + len(characters) + len(word_lengths):
                    if int(feat) in word_lengths_counts.keys():
                        vector[row][col] = word_lengths_counts.get(int(feat)) / text_size
                    else:
                        vector[row][col] = 0

                # Count digits
                elif col < len(LIWC) + len(characters) + len(word_lengths) + len(digits):
                    vector[row][col] = x_wo_punct.count(feat) / text_length

                # Count special symbols
                elif col < len(LIWC) + len(characters) + len(word_lengths) + len(digits) + len(symbols):
                    vector[row][col] = x.count(feat) / text_size

                # Count smileys
                elif col < len(LIWC) + len(characters) + len(word_lengths) + len(digits) + len(symbols) + len(smilies):
                    vector[row][col] = x.count(feat) / text_size
                    # print(feat, x.count(feat))

                # Count functions words
                elif col < len(LIWC) + len(characters) + len(word_lengths) + len(digits) + len(symbols) + len(smilies) + len(functions):
                    # vector[row][col] = len(re.findall(feat, " ".join(x).lower())) / text_size
                    vector[row][col] = sum(1 for i in re.finditer(feat, x_wo_punct)) / text_size

                # # Adding userId
                elif col < len(LIWC) + len(characters) + len(word_lengths) + len(digits) + len(symbols) + len(smilies) + len(functions) + len(user_id):
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

    characters, word_lengths, digits, symbols, smilies, functions, pos_tags, user_id, features, header_feature = FV_English_header()

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

            text_length = len(x)

            tmp_x = str.maketrans({key: None for key in string.punctuation})
            x_wo_punct = x.translate(tmp_x)

            x_words = nltk.word_tokenize(x_wo_punct)
            # print(x_words)
            pos = nltk.FreqDist([b for (a, b) in nltk.pos_tag(x_wo_punct)])

            word_lengths_counts = nltk.FreqDist([len(tok) for tok in x_words])

            for feat in features:
                if col < len(characters):
                    vector[row][col] = " ".join(x).count(feat) / text_length

                # Count word lengths
                elif col < len(characters) + len(word_lengths):
                    if int(feat) in word_lengths_counts.keys():
                        vector[row][col] = word_lengths_counts.get(int(feat)) / text_size
                    else:
                        vector[row][col] = 0

                # Count digits
                elif col < len(characters) + len(word_lengths) + len(digits):
                    vector[row][col] = x_wo_punct.count(feat) / text_length

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

def FV_Swedish_header():
    try:
        user_id = ['User_ID']

        word_lengths = [str(x) for x in list(range(1, 21))]
        characters = list('abcdefghijklmnopqrstuvwxyzåöä')
        digits = [str(x) for x in list(range(0, 10))]
        symbols = list('.?!,;:()"-\'')
        smileys = [':\')', ':-)', ';-)', ':p', ':d', ':x', '<3', ':)', ';)', ':@', ':*', ':j', ':$', '%)']
        functions = get_function_words(props.swe_function_word_filepath)

        tmp_LIWC_header = sorted(os.listdir(props.LIWC_filepath))

        LIWC_header = [x.replace(".txt","") for x in tmp_LIWC_header]

        digits_header = ['Digit_0', 'Digit_1', 'Digit_2', 'Digit_3', 'Digit_4', 'Digit_5', 'Digit_6', 'Digit_7',
                 'Digit_8', 'Digit_9']
        symbols_header = ['dot', 'question_mark', 'exclamation', 'comma', 'semi_colon', 'colon', 'left_bracket',
                  'right_bracket', 'double_inverted_comma', 'hypen', 'single_inverted_comma']
        smilies_header = ['smily_1', 'smily_2', 'smily_3', 'smily_4', 'smily_5', 'smily_6', 'smily_7', 'smily_8',
                  'smily_9', 'smily_10', 'smily_11', 'smily_12', 'smily_13', 'smily_14']

        header_feature = LIWC_header + characters + word_lengths + digits_header + symbols_header + smilies_header + \
                 functions + user_id


        features = tmp_LIWC_header + characters + word_lengths + digits + symbols + smileys + functions + user_id

    except Exception:
        traceback.print_exc()

    return tmp_LIWC_header, characters, word_lengths, digits, symbols_header, smilies_header, functions, user_id, features, header_feature


def FV_English_header():
    user_id = ['User_ID']

    word_lengths = [str(x) for x in list(range(1, 21))]
    characters = list('abcdefghijklmnopqrstuvwxyz')
    digits = [str(x) for x in list(range(0, 10))]
    symbols = list('.?!,;:()"-\'')
    smileys = [':\')', ':-)', ';-)', ':p', ':d', ':x', '<3', ':)', ';)', ':@', ':*', ':j', ':$', '%)']
    functions = get_function_words(props.eng_function_word_filepath)

    pos_tags = pos_header()

    digits_header = ['Digit_0', 'Digit_1', 'Digit_2', 'Digit_3', 'Digit_4', 'Digit_5', 'Digit_6', 'Digit_7',
             'Digit_8', 'Digit_9']
    symbols_header = ['dot', 'question_mark', 'exclamation', 'comma', 'semi_colon', 'colon', 'left_bracket',
              'right_bracket', 'double_inverted_comma', 'hypen', 'single_inverted_comma']
    smilies_header = ['smily_1', 'smily_2', 'smily_3', 'smily_4', 'smily_5', 'smily_6', 'smily_7', 'smily_8',
              'smily_9', 'smily_10', 'smily_11', 'smily_12', 'smily_13', 'smily_14']

    header_feature = characters + word_lengths + digits_header + symbols_header + smilies_header + \
             functions + pos_tags + user_id


    features = characters + word_lengths + digits + symbols + smileys + functions + pos_tags + user_id

    return characters, word_lengths, digits, symbols_header, smilies_header, functions, pos_tags, user_id, features, header_feature


def pos_header():
    header_wo_punct = []
    pos_tags = list(nltk.data.load('help/tagsets/upenn_tagset.pickle').keys())

    for single_pos_head in pos_tags:
        if single_pos_head.isalpha():
            header_wo_punct.append(single_pos_head)

    return header_wo_punct

def get_function_words(filepath):
    with open(filepath, 'r') as f:
        functions = [x.strip() for x in f.readlines()]

        for i in range(0, len(functions)):
            if len(re.findall('\(', functions[i])) == 1 and len(re.findall('\)', functions[i])) == 0:
                functions[i] = functions[i].replace('(', '\(')
            elif len(re.findall('\(', functions[i])) == 0 and len(re.findall('\)', functions[i])) == 1:
                functions[i] = functions[i].replace(')', '\)')
            if functions[i].endswith('*'):
                functions[i] = functions[i].replace('-*', '\\w*')
                functions[i] = '\\b' + functions[i]
            elif functions[i].startswith('*'):
                functions[i] = functions[i].replace('*-', '\\w*')
                functions[i] = '\\b' + functions[i]
            else:
                functions[i] = '\\b' + functions[i] + '\\b'
    return functions

def return_swe_result(x_test):
    # rf = joblib.load('static/model/swe_cal_rf_finalized_model.sav')
    rf = joblib.load(props.swedish_cal_rf_model_filename)
    predicted_test_scores = rf.predict_proba(x_test)

    # pred_class = float(rf.predict(x_test)[0])
    same_user_prob = str(predicted_test_scores[:, 0][0])
    diff_user_prob = str(predicted_test_scores[:, 1][0])

    return same_user_prob, diff_user_prob

def return_eng_result(x_test):
    # rf = joblib.load('static/model/eng_cal_rf_finalized_model.sav')
    rf = joblib.load(props.english_cal_rf_model_filename)
    predicted_test_scores = rf.predict_proba(x_test)

    # pred_class = float(rf.predict(x_test)[0])
    same_user_prob = str(predicted_test_scores[:, 0][0])
    diff_user_prob = str(predicted_test_scores[:, 1][0])

    return same_user_prob, diff_user_prob