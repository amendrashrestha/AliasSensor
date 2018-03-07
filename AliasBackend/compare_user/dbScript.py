__author__ = 'amendrashrestha'

from pymongo import MongoClient

import traceback

client = MongoClient('localhost', 27017) #dsg.foi.se
db = client.compare_user


def get_mongo_col():
    try:
        collection = db.collection_names()
    except Exception:
        traceback.print_exc()

    return collection


def get_user_post(collection_name, user, filed_id):
    try:
        collection = db[collection_name]
        # print(user)
        post_query = collection.find({filed_id: user}, {"text":1})
        posts_list = []

        for post in post_query:
            post = post['text']
            posts_list.append(post)
        return posts_list

    except Exception:
        traceback.print_exc()


def get_user_id(collection_name, id):
    try:
        collection = db[collection_name]
        admin_list = collection.distinct(id)
    except Exception:
        traceback.print_exc()

    return admin_list

def return_col_count(col_name):
    collection = db[col_name]
    return collection.find({}).count()

def get_user_gt_X(collection_name, field, threshold, limit):
    try:
        collection = db[collection_name]
        user_list = {}

        user_query = collection.aggregate([{'$match': {'Same': {'$gt': threshold}}}, {'$sort': {"count": -1}}, {"$limit": limit}])

        for single_info in user_query:
            user = single_info["User_B"]
            score = single_info["Same"]

            user_list[user] = score

        return user_list

    except Exception:
        traceback.print_exc()


def get_user_both(collection_name, user, filed_id):
    try:
        collection = db[collection_name]
        # print(user)
        post_query = collection.find({filed_id: user})
        posts_list = []

        for post in post_query:
            post = post['source']
            if post not in posts_list:
                posts_list.append(post)

        return ''.join(posts_list)

    except Exception:
        traceback.print_exc()

def get_user_flashback_both(collection_name, user, filed_id):
    client_dsg = MongoClient('dsg.foi.se', 27017)  # dsg.foi.se
    db_dsg = client_dsg.flashback

    try:
        collection = db_dsg[collection_name]
        # print(user)
        post_query = collection.find({filed_id: user}, {"category": 1})
        posts_list = []

        for post in post_query:
            post = post['category']
            if post not in posts_list:
                posts_list.append(post)

        return ''.join(posts_list)

    except Exception:
        traceback.print_exc()









