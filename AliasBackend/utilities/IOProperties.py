__author__ = 'amendrashrestha'
import os

LIWC_filepath = os.environ['HOME'] + '/repo/AliasSensor/AliasBackend/dictionaries/LIWC/'
function_word_filepath = os.environ['HOME'] + '/repo/AliasSensor/AliasBackend/dictionaries/function_words/funct.txt'

feature_vector_filepath = os.path.expanduser('~') + "/Desktop/AliasMatching/FV/feature_vector.csv" #feature_vector FV_Sample
feature_vector_sample_filepath = os.path.expanduser('~') + "/Desktop/AliasMatching/FV/FV_Sample.csv" #feature_vector FV_Sample


svm_model_filename = os.path.expanduser('~') + '/repo/AliasSensor/AliasBackend/models/svm_finalized_model.sav'
rf_model_filename = os.path.expanduser('~') + '/repo/AliasSensor/AliasBackend/models/rf_finalized_model.sav'
cal_svm_model_filename = os.path.expanduser('~') + '/repo/AliasSensor/AliasBackend/models/cal_svm_finalized_model.sav'

