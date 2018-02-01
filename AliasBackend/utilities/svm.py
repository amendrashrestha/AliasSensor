import os
import time
import sys

import numpy as np

from nltk.tokenize import TweetTokenizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler

sys.path.append(os.environ['HOME'] + "/PycharmProjects/exjobb")
from test_evaluator import evaluator
from data_loader import load_dataset
from transformers import PosTransformer, StyleTransformer, TokenizerTransformer, BeautifyTransformer

np.random.seed(7)

def testing_model(x_train, x_test, y_train, y_test, kfold, percentile):
    start_train = time.time()
    penalty = 'l2'
    loss = 'squared_hinge'

    svc = LinearSVC(C=1.0, penalty=penalty, loss=loss, dual=True)
    if kfold == 1:
        kfold = ShuffleSplit(test_size=0.20, n_iter=1, random_state=0)

    pipeline = Pipeline([
        ('featureselection', SelectPercentile(chi2, percentile=percentile)),
        ('model', svc)
    ])

    params = dict(model__C=[2 ** -12, 2 ** -9, 2 ** -7, 2 ** -5, 2 ** -3, 2 ** -1, 2 ** 1])
    gs = GridSearchCV(estimator=pipeline, cv=kfold, param_grid=params)
    print('\nTraining model with', str(percentile), "% features")
    gs.fit(x_train, y_train)
    best_est = gs.best_estimator_
    print('Time training:', time.time() - start_train)

    # logger_params = str(percentile) + '_' + str(kfold) + '_' + str(min_df) + '_' + str(max_df)
    logger_params = str(percentile) + '_' + str(kfold)
    # svm_training_<n_features>_<kfold>_<min_df>_<max_df>.log
    training_logger_name = os.environ['HOME'] + '/svm_training_' + logger_params + '.log'
    # svm_testing_<n_features>_<kfold>_<min_df>_<max_df>.log
    testing_logger_name = os.environ['HOME'] + '/svm_testing_' + logger_params + '.log'

    with open(training_logger_name, 'w') as f:
        f.write('Result from grid search cross validation:\n')
        for k in gs.cv_results_.keys():
            f.write(k + "\t" + str(gs.cv_results_[k]) + '\n')
        f.write('\nBest estimator parameters:\n')
        f.write(str(best_est.get_params()))

    print('Testing model')
    # Save the classifiers predictions and create confusion matrix
    y_test_pred = best_est.predict(x_test)
    duration = time.time() - start_train
    evaluator(testing_logger_name, [1, 2], [1, 2], y_test, y_test_pred, duration)

    accuracy = best_est.score(x_test, y_test)
    print("Accuracy: %.2f%%" % (accuracy * 100))


def main(train_dataset='pan17', test_dataset=None, corpus_types=None, n_files=0, kfold=5):
    # Read and preprocess input data
    print('Loading data from csv file')
    xx_train, y_train, xx_test, y_test = load_dataset(train_dataset, test_dataset, n_files, corpus_types)

    hyperparameters = [
        (0.01, 0.9),
        (0.1, 0.8),
        (0.001, 0.8),
        (0.001, 0.9)
    ]

    for mindf, maxdf in hyperparameters:
        print("Starting test with min_df=", str(mindf), "and max_df=", str(maxdf))
        pipeline = Pipeline([
            ('remove-html', BeautifyTransformer()),
            ('start', FeatureUnion([
                ('tfidfpipe', Pipeline([
                    ('tfidf', TfidfVectorizer(input='content', tokenizer=TweetTokenizer().tokenize, sublinear_tf=True, min_df=mindf, max_df=maxdf)),
                    ('tf-standardization', StandardScaler(copy=True, with_mean=False, with_std=True)),
                ])),
                ('tokenized-pipe', Pipeline([
                    ('tokenize', TokenizerTransformer()),
                    ('feature-extraction', FeatureUnion([
                        ('pos-pipe', Pipeline([
                            ('pos', PosTransformer()),
                            ('pos-standardization', StandardScaler(copy=True, with_mean=False, with_std=True))
                        ])),
                ('style-pipe', Pipeline([
                    ('style', StyleTransformer()),
                    ('style-standardization', StandardScaler(copy=True, with_mean=False, with_std=True))
                ]))
                ]))
                ]))
            ]))])

    x_train = pipeline.fit_transform(xx_train, y_train)
    x_test = pipeline.transform(xx_test)

    print("Preprocessing finished!")
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    for p in percentiles:
        testing_model(x_train, x_test, y_train, y_test, kfold, p, mindf, maxdf)
        # testing_model(x_train, x_test, y_train, y_test, kfold, p)


if __name__ == '__main__':
    main()
