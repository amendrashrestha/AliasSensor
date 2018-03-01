__author__ = 'amendrashrestha'

import os
import traceback
import warnings
from tqdm import tqdm

import sys

warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.environ['HOME'] , 'repo/AliasSensor/AliasBackend/utilities/'))
sys.path.append(os.path.join(os.environ['HOME'] , 'repo/AliasSensor/AliasBackend/model/'))

import pandas as pd

import IOReadWrite as IO
import dbScript as db

user_A_json_filepath = os.path.join(os.environ['HOME'] , "Desktop/File/Data/tomazin")

user_prob_filepath = os.path.join(os.environ['HOME'] , "Desktop/File/Result/")

def init():
    # compare_with_json_user()
    # compare_with_flashback_user()
    # get_collection_count()
    get_filtered_user_post()

def get_filtered_user_post():
    filtered_user_text_filepath = os.path.join(os.environ['HOME'], "Desktop/File/filtered_users.tsv")

    users = db.get_user_gt_ninety('result', 'User_B')

    for single_user, score in users.items():

        print('%s --> %d' %(single_user, score))

        source = db.get_user_both('data_all', single_user, 'user_id')

        if (len(source) > 0):
            # print(post)
            tmp_list = [single_user, score, source]
            IO.write_in_file(filtered_user_text_filepath, tmp_list)
            print("\n")

    for single_user in users:
        source = db.get_user_flashback_both('post_compare_user', single_user, 'user')
        if (len(source) > 0):
            # print(single_user)
            # print(post)
            tmp_list = [single_user, score, source]
            IO.write_in_file(filtered_user_text_filepath, tmp_list)
            print("\n")



def get_collection_count():
    collection_name = db.get_mongo_col()

    for single_col in collection_name:
        if "system.indexes" not in single_col:
            col_count = db.return_col_count(single_col)
            print('%s --> %d' %(single_col, col_count))

def compare_with_json_user():
    collection_name = db.get_mongo_col()

    for single_col in collection_name:
        if "system.indexes" not in single_col:
        # print(single_col)
            col_name, user_A_info, user_B_info = get_user_text(single_col)
            compare_user(col_name, user_A_info, user_B_info)

def compare_with_flashback_user():
    single_col = "post_compare_user"
    col_name, user_A_info, user_B_info = get_user_text(single_col)
    compare_user(col_name, user_A_info, user_B_info)


def get_user_text(collection_name):
    print(collection_name)
    id = "user" #admin_id

    user_A_info = {}
    user_B_info = {}

    try:
        user_A_post = IO.read_text_file(user_A_json_filepath)
        user_B = db.get_user_id(collection_name, id)

        for single_user in user_B:
            if single_user is not None and single_user != '':
                # print(single_user)
            # single_user = "darknesss"
                posts = db.get_user_post(collection_name, single_user, id)
                user_B_info[single_user] = posts

        user_A_info['tomazin'] = user_A_post

        return collection_name, user_A_info, user_B_info

    except Exception:
        traceback.print_exc()

def compare_user(col_name, user_a_list, user_b_list):
    header_feature = ['User_A','User_B', 'Same','Diff']

    filepath = user_prob_filepath + col_name + ".tsv"

    if not os.path.exists(filepath):
        IO.create_file_with_header(filepath, header_feature)

    # compare_same_list_users(user_a_list)
    # compare_same_list_users(user_b_list)
    compare_diff_list_users(user_a_list, user_b_list, filepath)

def compare_same_list_users(user_list):
    for i in range(0, len(user_list)-1):
        for j in range(1, len(user_list)):
            if i != j:
                user_a = list(user_list.keys())[i]
                user_b = list(user_list.keys())[j]

                text1 = user_list[list(user_list.keys())[i]]
                text2 = user_list[list(user_list.keys())[j]]

                create_file_with_prob(user_a, user_b, text1, text2, "same_list")

    print("------------------")

def compare_diff_list_users(user_a_list, user_b_list, filepath):
    for single_user_a in user_a_list:
        for single_user_b in tqdm(user_b_list):
            # print(single_user_a + " --> " + single_user_b)
            # print(single_user_b)
            text1 = user_a_list[single_user_a]
            text2 = user_b_list[single_user_b]

            create_file_with_prob(single_user_a, single_user_b, text1, text2, filepath)

def create_file_with_prob(user_a, user_b, text1, text2, filepath):
    try:
        post_A = " ".join(str(x) for x in text1)
        post_B = " ".join(str(x) for x in text2)

        fv_dataframe = IO.create_swedish_feature_vector(post_A, post_B)

        df = pd.DataFrame(fv_dataframe)
        # print(df)
        abs_fv = abs(df.diff()).dropna()

        x_test = abs_fv.iloc[:,0:len(abs_fv.columns)-1]

        same_user_prob, diff_user_prob = IO.return_swe_result(x_test)

        info = [user_a, user_b, same_user_prob, diff_user_prob]

        IO.write_in_file(filepath, info)

    except Exception:
        traceback.print_exc()

if __name__ == '__main__':
    init()

