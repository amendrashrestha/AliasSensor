__author__ = 'amendrashrestha'
import random
import re
import os

import AliasBackend.utilities.IOProperties as props


#randomize the list and split into 2 halves
def split_list(arr):
    # print(len(arr))
    random.shuffle(arr)
    half = len(arr)//2
    return arr[:half], arr[half:]

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

def create_file_with_header(filepath, features):
    with open(filepath, 'a') as outcsv:
        features = ','.join(features)
        features = features.replace("\\b", "").replace("\w", "").replace("*","")
        outcsv.write(features)
        outcsv.write("\n")

def FV_header():
    user_id = ['User_ID']

    word_lengths = [str(x) for x in list(range(1, 21))]
    digits = [str(x) for x in list(range(0, 10))]
    symbols = list('.?!,;:()"-\'')
    smileys = [':\')', ':-)', ';-)', ':p', ':d', ':x', '<3', ':)', ';)', ':@', ':*', ':j', ':$', '%)']
    functions = get_function_words(props.function_word_filepath)
    # tfidf = utilities.get_wordlist(props.tfidf_filepath)
    # ngram_char = utilities.get_wordlist(props.ngram_char_filepath)

    tmp_LIWC_header = sorted(os.listdir(props.LIWC_filepath))

    LIWC_header = []

    for single_LIWC_header in tmp_LIWC_header:
        LIWC_header.append(single_LIWC_header.replace(".txt",""))

    digits_header = ['Digit_0', 'Digit_1', 'Digit_2', 'Digit_3', 'Digit_4', 'Digit_5', 'Digit_6', 'Digit_7',
             'Digit_8', 'Digit_9']
    symbols_header = ['dot', 'question_mark', 'exclamation', 'comma', 'semi_colon', 'colon', 'left_bracket',
              'right_bracket', 'double_inverted_comma', 'hypen', 'single_inverted_comma']
    smilies_header = ['smily_1', 'smily_2', 'smily_3', 'smily_4', 'smily_5', 'smily_6', 'smily_7', 'smily_8',
              'smily_9', 'smily_10', 'smily_11', 'smily_12', 'smily_13', 'smily_14']
    # ngaram_char_header = utilities.create_ngram_header(ngram_char)

    # header_feature = LIWC_header + lengths + word_lengths + digits_header + symbols_header + smilies_header + \
    # functions + tfidf + ngaram_char_header + user_id

    # features = LIWC_header + lengths + word_lengths + digits + symbols + smileys + functions + tfidf + \
    # ngram_char + user_id

    header_feature = LIWC_header + word_lengths + digits_header + symbols_header + smilies_header + \
             functions + user_id


    features = tmp_LIWC_header +  word_lengths + digits + symbols + smileys + functions + user_id

    return tmp_LIWC_header, word_lengths, digits, symbols_header, smilies_header, functions, user_id, features, header_feature