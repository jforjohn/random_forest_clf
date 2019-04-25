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
from time import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    ##
    # Loads config
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", default="random_forest.cfg",
        help="specify the location of the clustering config file"
    )
    args, _ = parser.parse_known_args()

    config_file = args.config
    config = load(config_file, args)

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
            df , labels, train_size=test_percentage, random_state=42)
    clf = RF(
        f=f,
        nt=nt,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        seed=seed
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print(accuracy_score(y_test, pred))
    ##

