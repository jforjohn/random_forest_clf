##
from scipy.io.arff import loadarff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class MyPreprocessing:
    def __init__(self, raw=False):
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)

    def fit(self, df):
        # get label
        labels = df.iloc[:, -1]
        self.labels_ = labels
        labels_fac = pd.Series(pd.factorize(labels)[0])
        self.labels_fac = labels_fac
        df = df.drop(df.columns[len(df.columns) - 1], axis=1)
        #nan_cols = df.loc[:, df.isna().any()].columns

        # normalize numerical data
        df_num = df.select_dtypes(exclude=['object', 'bool'])
        df_obj = df.select_dtypes(include=['object', 'bool'])
        df_obj = df_obj.astype('object')
        #if df_num.size > 0:
        df_num = df_num.fillna(df_num.mean())
        df_num = pd.DataFrame(df_num, columns=df_num.columns)
        #else:
        #    df_normalized = pd.DataFrame()

        #if df_obj.size > 0:
        df_obj = df_obj.fillna('missing42')
        booleanDictionary = {True: 'TRUE', False: 'FALSE'}
        df_obj = df_obj.replace(booleanDictionary)
        '''
            col_2vals = np.array([df_obj[col].unique().size==2 for col in df_obj.columns])
            df_obj_2val = df_obj.iloc[:, col_2vals]
            df_obj_vals = df_obj.iloc[:, ~col_2vals]
            if df_obj_2val.empty:
                df_encoded_2val = pd.DataFrame()
            else:
                df_encoded_2val = pd.get_dummies(df_obj_2val,
                                    drop_first=True)
            if df_obj_vals.empty:
                df_encoded_vals = pd.DataFrame()
            else:
                df_encoded_vals = pd.get_dummies(df_obj_vals)

            df_encoded = pd.concat(
                [df_encoded_2val, df_encoded_vals],
                axis=1,
                sort=False)

        else:
            df_encoded = pd.DataFrame()
        '''
        self.new_df = pd.concat(
            [df_num, df_obj], #, labels_fac],
            axis=1,
            sort=False)
