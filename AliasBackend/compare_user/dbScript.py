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
        post_query = collection.find({filed_id: user}, {"content":1})
        posts_list = []

        for post in post_query:
            post = post['content']
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




