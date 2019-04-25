##
from MyPreprocessing import MyPreprocessing
import pandas as pd
import numpy as np
from config_loader import load
import argparse
import sys
from os import path

from MyRandomForest import MyRandomForest as RF
from time import time
from sklearn.model_selection import KFold
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

    kfold = config.getint('rf', 'kfold')
    nt = config.getint('rf', 'nt')
    f = config.getint('rf', 'f')
    max_depth = config.getint('tree', 'max_depth')
    min_samples_leaf = config.getint('tree', 'min_samples_leaf')

    # Preprocessing
    preprocess = MyPreprocessing()
    preprocess.fit(df_dataset)
    df = preprocess.new_df
    #labels = preprocess.labels_
    labels = preprocess.labels_
    #print(df.head())
    #print(labels.head())

    kf = KFold(n_splits=kfold)

    ## remove break
    for train_index, test_index in kf.split(df):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = df.iloc[train_index,:], df.iloc[test_index, :]
        y_train, y_test = labels[train_index], labels[test_index]
        clf = RF(
            f=f,
            nt=nt,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth
        )
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        print(pred.shape, y_test.shape)
        print(accuracy_score(y_test, pred))
        ##
        break

