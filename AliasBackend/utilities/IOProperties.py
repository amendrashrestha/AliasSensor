__author__ = 'amendrashrestha'
import os

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

LIWC_New_filepath = os.path.join(os.environ['HOME'] , '/repo/AliasSensor/AliasBackend/dictionaries/LIWC_New/')
LIWC_filepath = os.path.join(os.environ['HOME'] , '/repo/AliasSensor/AliasBackend/dictionaries/LIWC/')

swe_function_word_filepath = os.path.join(os.environ['HOME'] , 'repo/AliasSensor/AliasBackend/dictionaries/function_words/swe_funct.txt')
eng_function_word_filepath = os.path.join(os.environ['HOME'] , 'repo/AliasSensor/AliasBackend/dictionaries/function_words/eng_funct.txt')

# swedish_feature_vector_filepath = os.path.expanduser('~') + "/Desktop/AliasMatching/FV/swedish_feature_vector_flashback.csv" #feature_vector FV_Sample
# englsih_feature_vector_filepath = os.path.expanduser('~') + "/Desktop/AliasMatching/FV/englsih_feature_vector_reddit.csv" #englsih_feature_vector_reddit.csv

swedish_feature_vector_filepath = os.path.join(os.environ['HOME'], 'Desktop/AliasMatching/FV/swedish_feature_vector_flashback_2013.csv') #feature_vector FV_Sample
englsih_feature_vector_filepath = os.path.join(os.environ['HOME'], 'Desktop/AliasMatching/FV/englsih_feature_vector_reddit.csv') #englsih_feature_vector_reddit.csv

svm_model_filename = os.path.expanduser('~') + '/repo/AliasSensor/AliasBackend/models/svm_finalized_model.sav'
rf_model_filename = os.path.expanduser('~') + '/repo/AliasSensor/AliasBackend/models/rf_finalized_model.sav'

swedish_cal_svm_model_filename = os.path.expanduser('~') + '/repo/AliasSensor/AliasBackend/models/swe_cal_svm_finalized_model.sav'
english_cal_svm_model_filename = os.path.expanduser('~') + '/repo/AliasSensor/AliasBackend/models/eng_cal_svm_finalized_model.sav'

english_cal_rf_model_filename = os.path.join(os.environ['HOME'],'Desktop/AliasMatching/model/eng_cal_rf_finalized_model.sav')
swedish_cal_rf_model_filename = os.path.join(os.environ['HOME'], 'Desktop/AliasMatching/model/swe_cal_rf_finalized_model.sav')


