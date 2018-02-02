__author__ = 'amendrashrestha'

import sys
import os

sys.path.append(os.environ['HOME'] + "/repo/AliasSensor/")

from AliasBackend.main.controller import classification, init_swedish, init_english
from AliasBackend.main.classifiers import calibratedClassification

def main():
    init_english()
    # init_swedish()
    # classification()
    # calibratedClassification()

if __name__ == "__main__":
    main()
