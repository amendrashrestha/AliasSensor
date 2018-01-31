__author__ = 'amendrashrestha'

import os
import re
import string
import traceback

import numpy as np
import nltk

import AliasBackend.utilities.IOProperties as props
import AliasBackend.utilities.IOReadWrite as IO


class EnglishStyloFeatures():
    def __init__(self, user, posts):
        self.transform(user, posts)

    def transform(self, user, posts):
        # print(user)

        characters, word_lengths, digits, symbols, smilies, functions, pos_tags, user_id, features, header_feature = IO.FV_English_header()

        if not os.path.exists(props.englsih_feature_vector_filepath):
            IO.create_file_with_header(props.englsih_feature_vector_filepath, header_feature)

        post_A, post_B = IO.split_list(posts)

        post_A = " ".join(str(x) for x in post_A)
        post_B = " ".join(str(x) for x in post_B)

        post_list = []
        post_list.append(post_A)
        post_list.append(post_B)

        # post_list.append("makare-basd aldrig a :-) kvartsij bäst bh abc-makare:') ")
        # post_list.append(".. i vilket huvudfe fall som helst :D försöker ?? .")

        row = 0
        col = 0

        for x in post_list:
            # x = "länka till reportage ur massmedia. Det ska då vara sakliga reportage med integrations- och invandringspolitiska teman. @ 02:30-03:25. "
            # x = "This is a foo bar sentence."
            vector = np.zeros((1, len(features)))
            x = x.lower()
            split_text = x.split()
            text_size = len(split_text)

            tmp_x = str.maketrans({key: None for key in string.punctuation})
            x_wo_punct = x.translate(tmp_x)

            x_words = nltk.word_tokenize(x_wo_punct)
            # print(x_words)
            pos = nltk.FreqDist([b for (a, b) in nltk.pos_tag(x_wo_punct)])

            word_lengths_counts = nltk.FreqDist([len(tok) for tok in x_words])

            # for word, count in word_lengths_counts.most_common().sort():
            #     print(word, count)

            for feat in features:
                try:
                    # Count english alphabates
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

                except Exception:
                    traceback.print_exc()

            with open(props.englsih_feature_vector_filepath, 'ab') as f_handle:
                np.savetxt(f_handle, vector, delimiter=",")

