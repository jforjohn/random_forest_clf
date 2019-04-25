from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from scipy.stats import mode
import random
from MyDecisionTree import CART
from multiprocessing import Process
from queue import Queue

class MyRandomForest:
  def __init__(self,
               f, max_depth,
               min_samples_leaf, nt,
               seed
              ):
    #self.n_samples = n_samples
    # n_features
    self.f = f
    self.max_depth = max_depth
    self.min_samples_leaf = min_samples_leaf
    # number of trees
    self.nt = nt
    self.seed = seed
    random.seed(self.seed)
  
  def fit(self, X, y):
    n = X.shape[0]
    models = np.array([])
    feature_importance = []
    #mp_processes = []
    q = Queue()
    for ind in range(self.nt):
      print('Estimator', ind)
      randlist = pd.DataFrame(index=random.sample(X.index.tolist(),n))
      df_bootstap = X.merge(randlist, left_index=True, right_index=True, how='right')
      labels = pd.DataFrame(y).merge(randlist, left_index=True, right_index=True, how='right')
      labels = labels.iloc[:,0]
      #features = random.sample(X.columns.tolist(), self.f)
      #df = df_bootstap[features]
      df = df_bootstap
      clf = CART(
        max_depth=self.max_depth,
        min_samples_leaf=self.min_samples_leaf,
        f=self.f,
        random_forest=True,
        seed=self.seed
      )
      clf.fit(df, labels)
      feature_importance.append(clf.feature_importance)
      models = np.append(models, clf)
      #print(clf.print_tree())
      '''
      mp_processes.append(
        Process(target=clf.fit,
                   args=(df, labels))
      )
    for p in mp_processes:
      p.start()

    for p in mp_processes:
      p.join()
    
    for p in mp_processes:
      clf = q.get()
    '''
    self.models = models
    self.feature_importance = np.vstack(feature_importance)
    return self

  def predict(self, X):
    preds = []
    for clf in self.models:
      dt_preds = clf.predict(X)
      preds.append(dt_preds)
    predictions = mode(np.vstack((preds, dt_preds)))[0]
    return predictions.reshape(-1,1)

      
