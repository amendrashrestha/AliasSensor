__author__ = 'amendrashrestha'

from pymongo import MongoClient

import traceback

client = MongoClient('localhost', 27017) #dsg.foi.se
db = client.reddit
user_collection = db['username']
post_collection = db['posts']

collection = db['posts']

# returns user having post count greater than 10 and less than 50
def get_users():
    try:
        # user_query = collection.aggregate([{"$group": {'_id':"$user", 'count':{'$sum':1}}}, {'$match': {'count': {'$gt': 10, '$lt' : 50}}}])#, { "$limit": 2 }, {'$sort':{"count": -1}}
        user_query = user_collection.find({},{"_id":1})
        user_list = []

        for user in user_query:
            if user is not None:
                user = user['_id']
                user_list.append(user)
    except Exception:
        traceback.print_exc()

    return user_list

def get_user_post(user):
    try:
        # print(user)
        post_query = post_collection.find({"username": user}, {"text":1})
        posts_list = []

        for post in post_query:
            post = post['text']
            posts_list.append(post)
        return posts_list

    except Exception:
        traceback.print_exc()

