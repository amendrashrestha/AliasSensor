__author__ = 'amendrashrestha'

import os.path
import traceback
import sqlite3

from pymongo.mongo_client import MongoClient
mongo_object = MongoClient('localhost', 27017)


archive_filepath = os.path.expanduser('~') + "/Downloads/database.sqlite"

def get_all_post():

    try:
        conn = sqlite3.connect(archive_filepath)
        conn.row_factory = sqlite3.Row
        db = conn.cursor()

        rows = db.execute("select * from May2015").fetchall()

        conn.commit()
        conn.close()

        for single_row in rows:
            id = single_row['id']
            username = single_row['author']
            text = single_row['body']
            date = single_row['created_utc']

            db = mongo_object['reddit']
            db['posts'].insert({"id": id, "username": username, "date":date, "text": text})

    except exec():
        traceback.print_exc()

get_all_post()

