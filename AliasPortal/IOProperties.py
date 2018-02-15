__author__ = 'amendrashrestha'

import os

APP_ROOT = os.path.dirname(os.path.abspath(__file__))   # refers to application_top

LIWC_filepath = os.path.join(APP_ROOT, 'static/dictionaries/LIWC_New/')

swe_function_word_filepath = os.path.join(APP_ROOT, 'static/dictionaries/function_words/swe_funct.txt')
eng_function_word_filepath = os.path.join(APP_ROOT, 'static/dictionaries/function_words/eng_funct.txt')


