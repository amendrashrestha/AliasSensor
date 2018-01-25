__author__ = 'amendrashrestha'

import sys
import os

sys.path.append(os.environ['HOME'] + "/repo/AliasSensor/")

from AliasBackend.main.controller import classification

def main():
    # init()
    classification()

if __name__ == "__main__":
    main()
