__author__ = 'amendrashrestha'

import os
import traceback
import json
import warnings
import tqdm

warnings.filterwarnings("ignore")

import pandas as pd

import AliasPortal.IOReadWrite as IO

user_A_json_filepath = os.environ['HOME'] + "/Desktop/File/User_A/*.json"
user_B_json_filepath = os.environ['HOME'] + "/Desktop/Files/User_B/*.json"

user_prob_filepath = os.environ['HOME'] + "/Desktop/Files/user_prob.tsv"

def init():
    user_A_info, user_B_info = get_user_text()
    compare_user(user_A_info, user_B_info)

def get_user_text():
    user_A_json_files = IO.get_list_files(user_A_json_filepath)
    user_B_json_files = IO.get_list_files(user_B_json_filepath)

    user_A_info = {}
    user_B_info = {}

    for x in user_A_json_files:
        try:
            # x = "/home/amendra/Desktop/Files/User_A/A_1.json"
            with open(x) as json_data:
                data = json.load(json_data)
                # print(data)

                for single_data in data:
                    tmp_user_A_info = {}
                    username = single_data["name"]
                    text = single_data["text"]

                    user_A_info[username] = text
                    tmp_user_A_info[username] = text

                    # user_A_info = merge_text(user_A_info)

        except Exception:
            traceback.print_exc()

    for y in user_B_json_files:
        try:
            with open(y) as json_data:
                data = json.load(json_data)
                # print(data)

                for single_data in data:
                    username = single_data["name"]
                    text = single_data["text"]
                    user_B_info[username] = text

        except exec():
            traceback.print_exc()

    return user_A_info, user_B_info

def compare_user(user_a_list, user_b_list):
    header_feature = ['User_A','User_B', 'Same','Diff', 'Type']

    if not os.path.exists(user_prob_filepath):
        IO.create_file_with_header(user_prob_filepath, header_feature)

    compare_same_list_users(user_a_list)
    compare_same_list_users(user_b_list)
    compare_diff_list_users(user_a_list, user_b_list)

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

def compare_diff_list_users(user_a_list, user_b_list):
    for single_user_a in user_a_list:
        for single_user_b in user_b_list:
            # print(single_user_a + " --> " + single_user_b)

            text1 = user_a_list[single_user_a]
            text2 = user_b_list[single_user_b]

            create_file_with_prob(single_user_a, single_user_b, text1, text2, "diff_list")

def create_file_with_prob(user_a, user_b, text1, text2, type):

    fv_dataframe = IO.create_swedish_feature_vector(text1, text2)

    df = pd.DataFrame(fv_dataframe)
    # print(df)
    abs_fv = abs(df.diff()).dropna()

    x_test = abs_fv.iloc[:,0:len(abs_fv.columns)-1]

    same_user_prob, diff_user_prob = IO.return_swe_result(x_test)

    info = [user_a, user_b, same_user_prob, diff_user_prob, type]

    IO.write_in_file(user_prob_filepath, info)

    # print(same_user_prob + " , " + diff_user_prob)

def merge_text(ls):
    print(ls)
    print(type(ls))
    for k, v in ls.items():
        if k in ls:
            ls[k].append(v)
    return ls

if __name__ == '__main__':
    init()

