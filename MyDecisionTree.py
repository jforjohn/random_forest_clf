import numpy as np
from itertools import combinations
from threading import Thread
from queue import Queue
import random
import sys

class CART(object):
    def __init__(self,
                max_depth=4,
                min_samples_leaf=2,
                f=1,
                random_forest=True,
                q=None,
                seed=42
                ):
        self.feature = ''
        self.label = None
        self.n_samples = None
        self.gain = None
        self.left = None
        self.right = None
        self.threshold = None
        self.feature_type = None
        self.depth = 0
        self.root = None

        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.f = f
        self.random_forest = random_forest
        self.q = q
        #self.seed = seed
        #random.seed(seed)

    def fit(self, X, y):
        feature_importance = {}
        self.root = CART(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            f=self.f,
            random_forest=self.random_forest,
            #seed=self.seed,
            q=self.q
        )
        split_ind = X.index
        self.root._grow_tree(X, y, split_ind, feature_importance)
        #self.root._prune(self.prune, self.max_depth, self.min_criterion, self.root.n_samples)
        self.feature_importance = feature_importance
        #self.q.put(self)
        return self

    def predict(self, X):
        return np.array([self.root._predict(row) for _, row in X.iterrows()])

    def print_tree(self):
        self.root._show_tree(0, start_cond='if ')

    def _grow_tree(self, X, y, split_ind, feature_importance):
        if self.feature is None:
            return

        if self.random_forest:
            data = X.loc[split_ind, random.sample(X.columns.tolist(), self.f)]
        else:
            data = X.loc[split_ind, :]

        target = y.loc[split_ind]
        self.n_samples = data.shape[0]
        #print(data.shape, self.depth)
    
        target_unique = target.unique()
        target_val_cnts = target.value_counts()
        if self.depth >= self.max_depth or self.n_samples <= self.min_samples_leaf:
            # last feature evaluated
            feature_importance[self.feature] = feature_importance.get(self.feature, self.n_samples)
            self.feature = None
            self.label = target_val_cnts.index[0]
            return

        if len(target_unique) == 1 or self.gain == 0.0:
            # last feature evaluated
            feature_importance[self.feature] = feature_importance.get(self.feature, 0) + self.n_samples
            self.feature = None
            self.label = target_val_cnts.index[0]
            return
        
        self.label = target_val_cnts.index[0]
        #max([(c, len(target[target == c])) for c in target_unique], key = lambda x : x[1])[0]

        impurity_node = self._calc_impurity(target, target_unique, target_val_cnts)
        
        df_num = data.select_dtypes(exclude='object')
        df_obj = data.select_dtypes(include='object')
        #best_gain_num, best_feature_num, best_threshold_num = self._numerical_split_value(df_num, target, impurity_node)
        #best_gain_cat, best_feature_cat, best_threshold_cat = self._categ_split_value(df_obj, target, impurity_node)
        q = Queue()
        th1 = Thread(
            target=self._numerical_split_value,
            args=(df_num, target, impurity_node, q)
        )
        th1.start()

        th2 = Thread(
            target=self._categ_split_value,
            args=(df_obj, target, impurity_node, q)
        )
        th2.start()
        #th1.join()
        #th2.join()
        #q.join()
        q_elements = [q.get(), q.get()]
        for q_element in q_elements:
            if q_element[0] == 'num':
                _, best_gain_num, best_feature_num, best_threshold_num = q_element
            else: 
                _, best_gain_cat, best_feature_cat, best_threshold_cat = q_element
        #print(th1.is_alve(), th2.is_alive())

        if best_gain_cat >= best_gain_num and best_feature_cat:
            self.gain = best_gain_cat
            self.feature = best_feature_cat
            self.threshold = best_threshold_cat
            self.feature_type = 'cat'
        else:
            self.gain = best_gain_num
            self.feature = best_feature_num
            self.threshold = best_threshold_num
            self.feature_type = 'num'

        feature_importance[self.feature] = feature_importance.get(self.feature, self.n_samples)
        #self._split_tree(X, y, data, target, split_ind, feature_importance)

        ## Split the tree
        #print(self.feature_type, self.feature, self.threshold)
        if self.feature_type == 'num':
            features_l = data[data[self.feature] <= self.threshold]
            #target_l = target[data[self.feature] <= self.threshold]

            features_r = data[data[self.feature] > self.threshold]
            #target_r = target[data[self.feature] > self.threshold]
        else:
            features_l = data[data[self.feature].isin(self.threshold[0])]
            #target_l = target[data[self.feature].isin(self.threshold[0])]

            features_r = data[data[self.feature].isin(self.threshold[1])]
            #target_r = target[data[self.feature].isin(self.threshold[1])]

        if features_l.shape[0] == 0:
            self.feature = None
            return
        
        self.left = CART(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            f=self.f,
            random_forest=self.random_forest,
            #seed=self.seed,
            q=self.q
        )
        self.left.depth = self.depth + 1
        self.left.gain = self.gain
        self.left.feature = self.feature
        #self.left.label = self.label
        #print('l', self.gain, self.feature, self.threshold, features_l.shape, data.shape)
        self.left._grow_tree(X, y, features_l.index, feature_importance)

        if features_r.shape[0] == 0:
            self.feature = None
            return
        
        self.right = CART(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            f=self.f,
            random_forest=self.random_forest,
            #seed=self.seed,
            q=self.q
        )
        self.right.depth = self.depth + 1
        self.right.gain = self.gain
        self.right.feature = self.feature
        #self.right.label = self.label
        #print('r', self.gain, self.feature, self.threshold, features_r.shape, data.shape)
        self.right._grow_tree(X, y, features_r.index, feature_importance)

    def _numerical_split_value(self, data, target, impurity_node, *q):
        if q:
            q = q[0]
        best_gain, best_feature, best_threshold = 0.0, None, None
        for col in data.columns: #range(data.shape[1]):

            feature_level = data[col].unique()
            thresholds = (feature_level[:-1] + feature_level[1:]) / 2.0

            for threshold in thresholds:
                target_l = target[data[col] <= threshold]
                target_l_unique = target_l.unique()
                target_l_val_cnts = target_l.value_counts()
                impurity_l = self._calc_impurity(target_l, target_l_unique, target_l_val_cnts)
                n_l = float(target_l.shape[0]) / self.n_samples

                target_r = target[data[col] > threshold]
                target_r_unique = target_r.unique()
                target_r_val_cnts = target_r.value_counts()
                impurity_r   = self._calc_impurity(target_r, target_r_unique, target_r_val_cnts)
                n_r = float(target_r.shape[0]) / self.n_samples

                impurity_gain = impurity_node - (n_l * impurity_l + n_r * impurity_r)
                if impurity_gain >= best_gain:
                    best_gain = impurity_gain
                    best_feature = col
                    best_threshold = threshold
        #return best_gain, best_feature, best_threshold
        q.put(('num', best_gain, best_feature, best_threshold))
        #q.task_done()
    
    def _feature_values_combo(self, feature_vals):
        len_feature_vals = len(feature_vals)

        combos_vals = [([i], list(set(feature_vals)-set([i]))) for i in feature_vals]
        comb_len = int(len_feature_vals/2 + 1)

        for ind, combo in enumerate([list(combinations(feature_vals, i)) for i in range(2,comb_len)]):
            if len_feature_vals%2 == 0 :
                combo_len = -int(len(combo)/2) if ind+3 == comb_len else None
            else:
                combo_len = None
            for lista in combo[:combo_len]:
                combos_vals.append((list(lista), list(set(feature_vals)-set(lista))))

        #print(combos_vals)
        #print(len(combos_vals))
        return combos_vals

    def _categ_split_value(self, data, target, impurity_node, *q):
        if q:
            q = q[0]
        best_gain, best_feature, best_threshold = 0.0, None, None
        for col in data.columns:
            feature_level = data[col].unique()
            features_combo = self._feature_values_combo(feature_level)
            for feature_l, feature_r in features_combo:
                target_l = target[data[col].isin(feature_l)]
                target_l_unique = target_l.unique()
                target_l_val_cnts = target_l.value_counts()
                impurity_l = self._calc_impurity(target_l, target_l_unique, target_l_val_cnts)
                n_l = float(target_l.shape[0]) / self.n_samples
            
                target_r = target[data[col].isin(feature_r)]
                target_r_unique = target_r.unique()
                target_r_val_cnts = target_r.value_counts()
                impurity_r   = self._calc_impurity(target_r, target_r_unique, target_r_val_cnts)
                n_r = float(target_r.shape[0]) / self.n_samples

                impurity_gain = impurity_node - (n_l * impurity_l + n_r * impurity_r)
                if impurity_gain >= best_gain:
                    best_gain = impurity_gain
                    best_feature = col
                    best_threshold = (feature_l, feature_r)
        #return best_gain, best_feature, best_threshold
        q.put(('cat', best_gain, best_feature, best_threshold))
        #q.task_done()

    def _calc_impurity(self, target, target_unique, target_val_cnts):
        #if criterion == 'gini':
        return 1.0 - ((target_val_cnts/target.shape[0])**2).sum()
        #1.0 - sum([(float(len(target[target == c])) / float(target.shape[0])) ** 2.0 for c in target_unique])
    '''
    else:
        entropy = 0.0
        for c in np.unique(target):
            p = float(len(target[target == c])) / target.shape[0]
            if p > 0.0:
                entropy -= p * np.log2(p)
        return entropy
    '''        

    def _prune(self, method, max_depth, min_criterion, n_samples):
        if self.feature is None:
            return

        self.left._prune(method, max_depth, min_criterion, n_samples)
        self.right._prune(method, max_depth, min_criterion, n_samples)

        pruning = False

        if method == 'impurity' and self.left.feature is None and self.right.feature is None: 
            if (self.gain * float(self.n_samples) / n_samples) < min_criterion:
                pruning = True
        elif method == 'depth' and self.depth >= max_depth:
            pruning = True

        if pruning is True:
            self.left = None
            self.right = None
            self.feature = None

    def _predict(self, d):
        if self.feature != None:
            if self.feature_type == 'num':
                cond = d[self.feature] <= self.threshold
            else:
                cond = d[self.feature] in self.threshold[0]
            
            if cond:
                return self.left._predict(d)
            else:
                return self.right._predict(d)
        else: 
            return self.label

    def _show_tree(self, depth, start_cond=''):
        base = '  ' * depth
        #base_cond = start_cond
        if self.feature != None:
            if start_cond:
                cond = ''
            else:
                cond = ' and '
            if self.feature_type == 'num':
                base_tmp = str(self.feature) + ' <= ' + str(self.threshold)
            else:
                base_tmp = str(self.feature) + ' l_in ' + str(self.threshold[0])
            print(base + start_cond + base_tmp + ': (depth:%s)' %(depth))
            #base_cond = base_cond + cond + base_tmp
            self.left._show_tree(depth+1, start_cond=' if ')

            if self.feature_type == 'num':
                base_tmp = str(self.feature) + ' > ' + str(self.threshold)
            else:
                base_tmp = str(self.feature) + ' r_in ' + str(self.threshold[1])
            print(base + ' elif ' + base_tmp + ': (depth:%s)' %(depth))
            #base_cond = base_cond + cond + base_tmp
            self.right._show_tree(depth+1, start_cond=' if ')
        else:
            cond = 'then '
            print(base + cond + '{value: ' + str(self.label) + ', samples: ' + str(self.n_samples) + '}')