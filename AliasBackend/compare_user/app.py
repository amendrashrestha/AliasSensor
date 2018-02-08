__author__ = 'amendrashrestha'

import os
import traceback
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

import pandas as pd

import AliasPortal.IOReadWrite as IO
import AliasBackend.compare_user.dbScript as db

user_A_json_filepath = os.environ['HOME'] + "/Desktop/File/User_A/mail"

user_prob_filepath = os.environ['HOME'] + "/Desktop/File/user_prob.tsv"

def init():
    user_A_info, user_B_info = get_user_text()
    compare_user(user_A_info, user_B_info)

def get_user_text():
    collection_name = 'meddelande' #  adminlogg
    id = "skickare_id" #admin_id

    user_A_info = {}
    user_B_info = {}

    try:

        user_A_post = IO.read_text_file(user_A_json_filepath)
        user_B = db.get_user_id(collection_name, id)

        for single_user in user_B:
            # print(single_user)
            # single_user = "darknesss"
            posts = db.get_user_post(collection_name, single_user, id)
            user_B_info[single_user] = posts

        user_A_info['mail'] = user_A_post

        return user_A_info, user_B_info

    except Exception:
        traceback.print_exc()

def compare_user(user_a_list, user_b_list):
    header_feature = ['User_A','User_B', 'Same','Diff', 'Type']

    if not os.path.exists(user_prob_filepath):
        IO.create_file_with_header(user_prob_filepath, header_feature)

    # compare_same_list_users(user_a_list)
    # compare_same_list_users(user_b_list)
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
        for single_user_b in tqdm(user_b_list):
            # print(single_user_a + " --> " + single_user_b)
            print(single_user_b)
            text1 = user_a_list[single_user_a]
            text2 = user_b_list[single_user_b]

            create_file_with_prob(single_user_a, single_user_b, text1, text2, "meddelande")

def create_file_with_prob(user_a, user_b, text1, text2, type):
    try:
        post_A = " ".join(str(x) for x in text1)
        post_B = " ".join(str(x) for x in text2)

        fv_dataframe = IO.create_swedish_feature_vector(post_A, post_B)

        df = pd.DataFrame(fv_dataframe)
        # print(df)
        abs_fv = abs(df.diff()).dropna()

        x_test = abs_fv.iloc[:,0:len(abs_fv.columns)-1]

        same_user_prob, diff_user_prob = IO.return_swe_result(x_test)

        info = [user_a, user_b, same_user_prob, diff_user_prob, type]

        IO.write_in_file(user_prob_filepath, info)

    except Exception:
        traceback.print_exc()

if __name__ == '__main__':
    init()

