from MainLauncher import main
import argparse
from config_loader import load
import sys
from os import mkdir, path
from math import log, sqrt
import pandas as pd
from collections import OrderedDict

def experiment(config, cols, output_dir):
  exp_f_name = ['1', '3', 'log2(M)+1', 'sqrt(M)'] *2
  exp_f = [1, 3, int(log(cols,2)+1), int(sqrt(cols))] *2
  exp_nt = [50] *4 +  [100] *4
  exp_mds = [3,4,5,10]
  dataset = config.get('rf', 'dataset')
  print(dataset)
  for md in exp_mds:
    config.set('tree', 'max_depth', str(md))
    for ind in range(len(exp_f_name)):
      output_file = f'{dataset}_{exp_f[ind]}_{exp_nt[ind]}_{md}.txt'
      config.set('rf', 'f', str(exp_f[ind]))
      config.set('rf', 'nt', str(exp_nt[ind]))
      sys.stdout = open(path.join(output_dir,output_file), 'w')
      acc_tr, acc_tst, acc_tree, acc_sklearn, feature_importance, duration = main(config)

      # write results
      dict_results = OrderedDict({
        'Out File': output_file,
        'Dataset': dataset,
        'NT': exp_nt[ind],
        'F val': exp_f[ind],
        'F': exp_f_name[ind],
        'Max Depth': md,
        'Min Samples Leaf': config.get('tree', 'min_samples_leaf'),
        'Acc tr': round(acc_tr*100,2),
        'Acc tst': round(acc_tst*100,2),
        'Acc tree': round(acc_tree*100,2),
        'Acc sklearn': round(acc_sklearn*100,2),
        'Time': round(duration,2)
      })
      df_results = pd.DataFrame([dict_results])
      df_results.to_csv('results-rf.csv', mode='a', header=False, index=False)
      pd.DataFrame(feature_importance).T.to_csv('results-feature-importance.csv', mode='a', header=True, index=False)

if __name__ == '__main__':
  # Loads config
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-c", "--config", default="random_forest.cfg",
      help="specify the location of the clustering config file"
  )
  args, _ = parser.parse_known_args()

  config_file = args.config
  config = load(config_file, args)

  # Dataset
  config.set('rf', 'dataset', 'lymphography.csv')
  #config.set('rf', 'dataset', 'breastcancer.csv')
  #config.set('rf', 'dataset', 'primarytumor.csv')

  dataset_dir = config.get('rf', 'dataset_dir')
  dataset = config.get('rf', 'dataset')
  path_data = path.join(dataset_dir, dataset)

  try:
      df_dataset = pd.read_csv(path_data, header=0)
  except FileNotFoundError:
      print("Dataset '%s' cannot be found in the path %s" %(dataset, path_data))
      sys.exit(1)

  cols = df_dataset.shape[1]

output_dir = 'results'
try:
  # Create target Directory
  mkdir(output_dir)
except FileExistsError:
  print("Directory " , output_dir,  " already exists")
  #main(config)

# Dataset

experiment(config, cols, output_dir)
