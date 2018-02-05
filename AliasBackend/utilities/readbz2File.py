__author__ = 'amendrashrestha'

import bz2
import json
import os.path
import traceback

from pymongo.mongo_client import MongoClient
mongo_object = MongoClient('localhost', 27017)

import AliasBackend.utilities.IOReadWrite as IO

archive_filepath = os.path.expanduser('~') + "/Downloads/Data/Reddit/"

file_path = IO.get_list_files(archive_filepath)

for single_file in file_path:
    print(single_file)
    bz_file = bz2.BZ2File(single_file, 'rb', 100000000)

    while True:
        try:
            line = bz_file.readline().decode('utf8')

            if len(line) == 0:
                break
            comment = json.loads(line)

            id = comment["id"]
            body = comment["body"]
            author = comment["author"]
            date = comment["created_utc"]

            db = mongo_object['reddit']
            db['posts'].insert({"id":id, "user": author, "date": date, "text": body})

        except exec():
            traceback.print_exc()

    bz_file.close()
