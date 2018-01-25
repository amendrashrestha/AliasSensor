__author__ = 'amendrashrestha'
import os

LIWC_filepath = os.environ['HOME'] + '/repo/AliasSensor/dictionaries/LIWC/'
function_word_filepath = os.environ['HOME'] + '/repo/AliasSensor/dictionaries/function_words/funct.txt'

feature_vector_filepath = os.path.expanduser('~') + "/Desktop/AliasMatching/FV/feature_vector.csv" #feature_vector FV_Sample
feature_vector_sample_filepath = os.path.expanduser('~') + "/Desktop/AliasMatching/FV/FV_Sample.csv" #feature_vector FV_Sample


model_filename = os.path.expanduser('~') + '/repo/AliasSensor/models/finalized_model.sav'
rf_model_filename = os.path.expanduser('~') + '/repo/AliasSensor/models/rf_finalized_model.sav'


