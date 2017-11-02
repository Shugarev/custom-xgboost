#https://www.kaggle.com/ekababisong/credit-card-fraud-detection-using-xgboost

# import packages

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import os

seed = 128

path = "/home/sergey/data/merchant-11/"

training_set = pd.read_csv(os.path.join(path,'train.csv')).drop(['id','date_only'], axis=1)
#training_set = pd.read_csv(os.path.join(path,'test.csv')).drop(['id'], axis=1)
test_set = pd.read_csv(os.path.join(path,'test.csv')).drop(['id','date_only'], axis=1)


X_train = training_set.iloc[:, 1:].values
Y_train = training_set.iloc[:, 0].values

X_test = test_set.iloc[:, 1:].values
Y_test = test_set.iloc[:, 0].values


# XGBoost CV model
model = XGBClassifier(max_depth=6,
                      learning_rate=0.1,
                      n_estimators=100, silent=True,
                      objective="binary:logistic",
                      nthread=-1, gamma=0.7, min_child_weight=10,
                      max_delta_step=0, subsample=1, colsample_bytree=0.95, colsample_bylevel=1,
                      reg_alpha=0.00002, reg_lambda=10, scale_pos_weight=1,
                      base_score=0.5, seed=0, missing=None)
# n_split - number folds,default =3
# random_state - seed
kfold = StratifiedKFold(n_splits=10, random_state=7)

# use area under the precision-recall curve to show classification accuracy
scoring = 'roc_auc'

results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring = scoring)

print( "AUC: %.3f (%.3f)" % (results.mean(), results.std()) )

probas1 = model.fit(X_train, Y_train).predict_proba(X_test)
df_probas1 = pd.DataFrame(probas1, columns=['p1', 'p2'])
df_probas1.to_csv(os.path.join(path,'result-test-4.csv'))



