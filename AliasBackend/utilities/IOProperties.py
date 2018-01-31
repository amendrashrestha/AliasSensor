__author__ = 'amendrashrestha'
import os

LIWC_filepath = os.environ['HOME'] + '/repo/AliasSensor/AliasBackend/dictionaries/LIWC/'
swe_function_word_filepath = os.environ['HOME'] + '/repo/AliasSensor/AliasBackend/dictionaries/function_words/swe_funct.txt'
eng_function_word_filepath = os.environ['HOME'] + '/repo/AliasSensor/AliasBackend/dictionaries/function_words/eng_funct.txt'

feature_vector_sample_filepath = os.path.expanduser('~') + "/Desktop/AliasMatching/FV/FV_Sample.csv" #feature_vector FV_Sample
englsih_feature_vector_filepath = os.path.expanduser('~') + "/Desktop/AliasMatching/FV/englsih_feature_vector.csv"

svm_model_filename = os.path.expanduser('~') + '/repo/AliasSensor/AliasBackend/models/svm_finalized_model.sav'
rf_model_filename = os.path.expanduser('~') + '/repo/AliasSensor/AliasBackend/models/rf_finalized_model.sav'

swedish_cal_svm_model_filename = os.path.expanduser('~') + '/repo/AliasSensor/AliasBackend/models/swe_cal_svm_finalized_model.sav'
english_cal_svm_model_filename = os.path.expanduser('~') + '/repo/AliasSensor/AliasBackend/models/eng_cal_svm_finalized_model.sav'


