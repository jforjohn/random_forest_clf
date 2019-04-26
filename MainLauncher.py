##
from MyPreprocessing import MyPreprocessing
import pandas as pd
import numpy as np
from config_loader import load
import argparse
import sys
from os import path
from random import randint

from MyRandomForest import MyRandomForest as RF
from MyDecisionTree import CART
from time import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier as RFsklearn

def main(config):
    '''
    # Loads config
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", default="random_forest.cfg",
        help="specify the location of the clustering config file"
    )
    args, _ = parser.parse_known_args()

    config_file = args.config
    config = load(config_file, args)
    '''
    dataset_dir = config.get('rf', 'dataset_dir')
    dataset = config.get('rf', 'dataset')

    path_data = path.join(dataset_dir, dataset)

    try:
        df_dataset = pd.read_csv(path_data, header=0)
    except FileNotFoundError:
        print("Dataset '%s' cannot be found in the path %s" %(dataset, path_data))
        sys.exit(1)

    nt = config.getint('rf', 'nt')
    f = config.getint('rf', 'f')
    max_depth = config.getint('tree', 'max_depth')
    min_samples_leaf = config.getint('tree', 'min_samples_leaf')
    seed = config.getint('rf', 'seed')
    test_percentage = config.getfloat('rf', 'test_percentage')

    if seed < 0:
        seed = randint(1,100)
    # Preprocessing
    preprocess = MyPreprocessing()
    preprocess.fit(df_dataset)
    df = preprocess.new_df
    #labels = preprocess.labels_
    labels = preprocess.labels_
    #print(df.head())
    #print(labels.head())


    try:
        X_train, X_test, y_train, y_test = train_test_split(
            df , labels, test_size=test_percentage, random_state=seed, stratify=labels.values)
    except ValueError:
        # for the case of the least populated class in y to have only 1 member
        X_train, X_test, y_train, y_test = train_test_split(
            df , labels, train_size=test_percentage, random_state=seed)
    clf = RF(
        f=f,
        nt=nt,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        seed=seed
    )
    start = time()
    clf.fit(X_train, y_train)
    duration = time() - start
    print('Train duration', duration)
    feature_importance = clf.feature_importance
    print('Feature_importance')
    print(feature_importance)
    pred = clf.predict(X_test)
    acc_test = accuracy_score(y_test, pred)
    print('Acc test', acc_test)
    pred = clf.predict(X_train)
    acc_train = accuracy_score(y_train, pred)
    print('Acc train', acc_train)

    # compare
    print('Compare with results of a single tree')
    clf_dt = CART(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        f=f,
        random_forest=False,
    )
    clf_dt.fit(X_train, y_train)
    pred = clf_dt.predict(X_test)
    acc_tree = accuracy_score(y_test, pred)
    print('Acc tree', acc_tree)

    # sklearn
    print("Compare with sklearn's RandomForest")
    preprocess = MyPreprocessing(one_hot=True)
    preprocess.fit(df_dataset)
    df = preprocess.new_df
    #labels = preprocess.labels_
    labels = preprocess.labels_fac

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            df , labels, test_size=test_percentage, random_state=seed, stratify=labels.values)
    except ValueError:
        # for the case of the least populated class in y to have only 1 member
        X_train, X_test, y_train, y_test = train_test_split(
            df , labels, train_size=test_percentage, random_state=seed)
    
    clf_sk = RFsklearn(
        max_features=f,
        n_estimators=nt,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
    )
    clf_sk.fit(X_train, y_train)
    pred = clf_sk.predict(X_test)
    acc_sklearn = accuracy_score(y_test, pred)
    print('Acc sklearn', acc_test)

    return acc_train, acc_test, acc_tree, acc_sklearn, feature_importance, duration
