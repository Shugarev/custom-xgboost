from graphviz import Digraph

#http://xgboost.readthedocs.io/en/latest/python/python_intro.html
#https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

import numpy as np
import xgboost as xgb
import pandas as pd
import os

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing

path = "/home/sergey/data/merchant-11/"
write_header = 0

max_depth        = 6
eta              = 0.2
silent           = 1#1
objective        = 'binary:logistic'
nthread          = 20,

eval_metric      = 'logloss'#[auc, logloss],

min_child_weight = 10
gamma            = 0
subsample        = 1.0
colsample_bytree = 1
alpha            = 2e-05
lambda1          = 10
scale_pos_weight = 1

num_boost_round = 20
booster = 'gbtree'

params = {'max_depth'       : max_depth,
          'eta'             : eta,
          'silent'          : silent,
          'objective'       : objective,
          #'nthread'        : nthread,#5,
          'eval_metric'     : eval_metric,
          'min_child_weight': min_child_weight,
          'gamma'           : gamma,
          'subsample'       : subsample,
          'colsample_bytree': colsample_bytree,
          'alpha'           : alpha,
          'lambda'          : lambda1,
          'scale_pos_weight': scale_pos_weight,
          'booster'         : booster
          }

stat = [1,5,10,20,50,80]

names = ''
for x in stat:
    names = names + str(x) + '%, '
names += ', '.join(params.keys()) +  ", num_boost_round \n"

if write_header == 1 :
  write_header = 0
  with open(path + 'result.csv', 'a') as file:
    file.writelines(names)
  file.close()

training_set = pd.read_csv(os.path.join(path,'train.csv')).drop(['id','date_only'], axis=1)
test_set = pd.read_csv(os.path.join(path,'test.csv')).drop(['id','date_only'], axis=1)


# add fraud rows
# training_set_good = training_set[training_set['status']==1]
# for i in range(15):
#   training_set = training_set.append(training_set_good)
# training_set = training_set.sample(frac=1).reset_index(drop=True)


# for column_name in training_set.columns:
#   max_column = max( training_set[column_name])
#   min_column = min( training_set[column_name])
#   mean_column = np.mean(training_set[column_name])
#   std_column =   np.std(training_set[column_name])
#   if (column_name != 'status'):
#     training_set[column_name] = (training_set[column_name] - mean_column)/ std_column
#     test_set[column_name] = (test_set[column_name] - mean_column) / std_column

All_cb_in_test  = test_set[test_set['status'] == 1].shape[0]
All_row_in_test = test_set.shape[0]

train = training_set.as_matrix()
test  =  test_set.as_matrix()
dtest  = xgb.DMatrix(data=test[:,1:], label=test[:,0], missing=-999.0)
label = train[:,0]




#for sklearn
X_train = training_set.iloc[:, 1:].values
Y_train = training_set.iloc[:, 0].values

X_test = test_set.iloc[:, 1:].values
Y_test = test_set.iloc[:, 0].values
sample_weight_sklern = [30 if x == 1 else 1 for x in label]

def run_xgboost_with_params(dtrain,params, num_boost_round = 20, a=97, b=3):
  bst = xgb.train(params, dtrain, num_boost_round=num_boost_round)#100


  ypred = bst.predict(data=dtest)#, ntree_limit=bst.best_ntree_limit)

  #Plotting
  #xgb.plot_tree(bst, num_trees=2)


  df_ypred = pd.DataFrame(ypred, columns=['pred'])

  df_ypred['status'] = test_set['status']
  #df_ypred = df_ypred.sort('pred',ascending=False)
  df_ypred = df_ypred.sort_values(by="pred",ascending=False )

  All_cb_in_test  = df_ypred[df_ypred['status'] == 1].shape[0]
  All_row_in_test = df_ypred.shape[0]

  str_res = ''
  val = ', '.join(str(x) for x in params.values()) + ', ' + str(num_boost_round)
  total = 0
  for row in stat:
    count    = round (row * All_row_in_test /100) + 1
    count_cb = df_ypred [:count][df_ypred ['status'] == 1].shape[0]
    res = round( 100 * count_cb/ All_cb_in_test,2)
    str_res += str(res) + ', '
    total += res
  str_res += val;
  # params weight
  str_res += ', ' + ', '.join(str(x) for x in [a, b, total])
  str_res += "\n"
  print( str_res)

  with open(path + 'result.csv', 'a') as file:
    file.writelines(str_res)
  file.close()
  #  df_filtered = df[df['column'] == value]

def run_xgboost_with_sklearn(X_train, Y_train,X_test,
                        params,
                        max_depth=6,  learning_rate=0.1,
                        n_estimators=100, silent=True,
                        objective="binary:logistic",
                        nthread=-1, gamma=0.7, min_child_weight=10,
                        max_delta_step=0, subsample=1, colsample_bytree=0.95, colsample_bylevel=1,
                        reg_alpha=0.00002, reg_lambda=10, scale_pos_weight=1,
                        base_score=0.5, seed=0, missing=None):

  model = XGBClassifier(max_depth=max_depth,
                        learning_rate=learning_rate,
                        n_estimators=n_estimators, silent=silent,
                        objective=objective,
                        nthread=-nthread, gamma=gamma, min_child_weight=min_child_weight,
                        max_delta_step=max_delta_step, subsample=subsample, colsample_bytree=colsample_bytree, colsample_bylevel=colsample_bylevel,
                        reg_alpha=reg_alpha, reg_lambda=reg_lambda, scale_pos_weight=scale_pos_weight,
                        base_score=base_score, seed=seed, missing=missing)
  probas = model.fit(X_train, Y_train,sample_weight=sample_weight_sklern).predict_proba(X_test)#
  df_probas = pd.DataFrame(probas, columns=['p1', 'p2'])
  df_probas['status'] = test_set['status']
  df_probas = df_probas.sort_values(by="p2", ascending=False)
  str_res = ''
  val = ', '.join(str(x) for x in params.values()) + ', ' + str(num_boost_round)
  for row in stat:
    count    = round (row * All_row_in_test /100) + 1
    count_cb = df_probas [:count][df_probas ['status'] == 1].shape[0]
    res = round( 100 * count_cb/ All_cb_in_test,2)
    str_res += str(res) + ', '

  str_res += val
  #params weight
  # str_res += ', '.join(str(x) for x in [a,b])
  str_res += "\n"
  with open(path + 'result.csv', 'a') as file:
    file.writelines(str_res)
  file.close()
#  pass




# eval_metrics = ['logloss','auc',['auc', 'logloss']]
eval_metrics = ['auc','logloss','mae','rmse','ndcg','map']
num_boost_rounds = [20,30]
max_depthes = [4,6,7,8]
etas  =       [0.05,0.1,0.2]#0.2, 0.3,0.4
min_child_weights = [7,8,9,10,11,12,13,]
objectives = ['reg:linear','binary:logistic']
boosters = ['dart','gblinear','gbtree']
#gammas = [0,0.1,0.2,0.3,0.4,0.5]
gammas = [0,1,2,3,5,10,20,45,100]
lambdas = [8.5,9,9.2,9.25,9.8,10,10.2,10.25,10.5,10.8,11,11.2,11.5,11.8,12]


for l in lambdas:
  params['lambda'] = l
  weigth = [97 if x == 1 else 3 for x in label]
  dtrain = xgb.DMatrix(data=train[:, 1:], label=label, missing=-999.0, weight=weigth)#
  run_xgboost_with_params(dtrain, params, num_boost_round)



# for min_child_weight in min_child_weights:
#   params['min_child_weight'] = min_child_weight
#   weigth = [97 if x == 1 else 3 for x in label]
#   dtrain = xgb.DMatrix(data=train[:, 1:], label=label, missing=-999.0, weight=weigth)
#   run_xgboost_with_params(dtrain, params, num_boost_round)

#weigth = None
# for a in range(95,105):
#   for b in range(1,7):
#     weigth = [a if x == 1 else b for x in label]
#     #params['booster'] = 'dart'
#     dtrain = xgb.DMatrix(data=train[:, 1:], label=label, missing=-999.0, weight=weigth)  # ,
#     run_xgboost_with_params(dtrain, params, num_boost_round,a,b)







# for eta in np.arange(0.19, 0.212, 0.01):
#   for num_boost_round in range(10, 32, 10):
#     weigth = [97 if x == 1 else 3 for x in label]
#     params['eta'] = eta
#     dtrain = xgb.DMatrix(data=train[:, 1:], label=label, missing=-999.0, weight=weigth)
#     run_xgboost_with_params(dtrain, params, num_boost_round)


# for num_boost_round in num_boost_rounds:
#   for max_depth in max_depthes:
#     for eval_metric in eval_metrics:
#       for a,b in [[97,3]]:#,[104,4],[95,6]
#         for eta in etas:
#           params['eta'] = eta
#           weigth = [a if x == 1 else b for x in label]
#           #params['booster'] = 'dart'
#           params['max_depth'] = max_depth
#           params['eval_metric'] = eval_metric




# run_xgboost_with_sklearn(X_train, Y_train,X_test,params,learning_rate=eta,gamma=0,min_child_weight=20)

# for max_depth in max_depthes:
#    for eval_metric in eval_metrics:
#     params['max_depth'] = max_depth
#     params['eval_metric'] = eval_metric
#     # params['objective'] = 'reg:linear'
#     print("max_depth = ", max_depth, " , eval=" ,eval_metric)



#for sklearn


# objectives = ['multi:softprob'] - do not true


# for objective1 in objectives:
#   params['objective'] = objective1
#   for eta in etas:
#     params['eta'] = eta
#
#     run_xgboost_with_sklearn(params,learning_rate=eta,gamma=0,min_child_weight=20, objective=objective1)


# for i in range(1,6,1):
#   run_xgboost_with_params(params, num_boost_round)

# for scale_pos_weight in range(1,100,3):
#   params['scale_pos_weight'] = scale_pos_weight
#   run_xgboost_with_params(params, num_boost_round)

# for min_child_weight in min_child_weights:
#   params

# for num_boost_round in num_boost_rounds:
#   for max_depth in max_depthes:
#     for eta in etas:
#       params['max_depth'] = max_depth
#       params['eta'] = eta
#       run_xgboost_with_params(params, num_boost_round)
#       print("num_boost_round = ",num_boost_round," ,max_depth=",max_depth,"  ,eta= ",eta  )
