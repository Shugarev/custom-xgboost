import numpy as np
import pandas as pd
import os
path = "/home/sergey/PycharmProjects/FirstTest/data/merchant-11"

# training_set = pd.read_csv(os.path.join(path,'train-custom-xgboost.csv'))
training_set = pd.read_csv(os.path.join(path,'train.csv'))
training_set = training_set.drop(['id','date_only'], axis=1)

lambda_1 = 1
nu_1 = 1
T1 = []
T2 = []
total_weight = 1


def label_g(row):
  if row['status'] == 1:
    return total_weight * 2 * (row['y'] -row['status'])
  return  2 * (row['y'] - row['status'])


def first_step(dt,y = None):
  n_row,n_col = dt.shape
  #print(n_row)
  columns = dt.columns
  columns = columns.drop('status')
  if y is None:
    y = np.full(n_row, 0)
    # y =  [1 if x == 1 else 0 for x in dt['status'] ]
    # if y[15] ==1 :
    #  y[15] =0
    # else:
    #  y[15] =1

  if 'id' not in columns:
    dt['id'] = [x for x in range(n_row)]
  dt['y'] = y

  result = []
  for col in columns:
    #print( col)
    values = dt[col].unique()
    values_sort = np.sort(values)
    for val1 in range(values_sort.size -1):
      val = values_sort[val1]
      dt1 = dt[dt[col] <= val ]
      dt2 = dt[dt[col] > val ]

      g1 = get_g(dt1)
      g2 = get_g(dt2)

      h1 = get_h(dt1)
      h2 = get_h(dt2)

      obj_0 = get_obj1(g1,h1,g2,h2)
      # obj_2 = get_obj2(g1, h1, g2, h2)

      w_1 =   get_w(g1, h1)
      w_2 =   get_w(g2, h2)

      result.append({'column_name':col,'column_val':val,'w1':w_1,'w2':w_2,'obj':obj_0})
      print(col," = ", val, " , obj=",obj_0 , " ,w1 =",w_1, ",w2=",w_2 )


      # print(values)

  res = {'min':{'id':0,'val':result[0]['obj']},'max':{'id':0,'val':result[0]['obj']}}
  for i in range(1,len(result)):
    if result[i]['obj'] > res['max']['val']:
      res['max']['id'] = i
      res['max']['val'] = result[i]['obj']
    elif result[i]['obj'] < res['min']['val']:
      res['min']['id'] = i
      res['min']['val'] = result[i]['obj']


  print("max")
  tmp = result[ res['max']['id']]
  print(tmp['column_name'], " = ", tmp['column_val'], " , obj=", tmp['obj'], " ,w1 =", tmp['w1'], ",w2=", tmp['w2'])

  print("min")
  tmp = result[res['min']['id']]
  print(tmp['column_name'], " = ", tmp['column_val'], " , obj=", tmp['obj'], " ,w1 =", tmp['w1'], ",w2=", tmp['w2'])


  col_name = tmp['column_name']
  val = tmp['column_val']

  dt1 = dt[dt[col_name]<= val]
  dt1['y'] = tmp['w1']

  dt2 = dt[dt[col_name] > val]
  dt2['y'] = tmp['w2']

  #dt1.to_csv(os.path.join(path,'dt1.csv'),index=False)
  #dt2.to_csv(os.path.join(path,'dt2.csv'),index=False)

  pass

def training_los(dt):
  sum =0
  n_row, n_col = dt.shape
  for i in range(n_row):
    tmp = dt['status'].iloc[i] - dt['y'].iloc[i]
    if dt['status'].iloc[i] == 1:
      sum += total_weight * tmp *tmp
    else:
      sum +=  tmp * tmp
  # print(sum)
  return sum

def get_g(dt):
  sum =0
  n_row, n_col = dt.shape
  g = 0
  for i in range(n_row):
    if dt['status'].iloc[i] == 1:
      g  += total_weight * 2 * ( dt['y'].iloc[i] - dt['status'].iloc[i])
    else:
      g += 2 *  (dt['y'].iloc[i] - dt['status'].iloc[i])
    print("i=",i,", g=",g)


  dt['g'] = dt.apply (lambda row: label_g (row),axis=1)
  g0 = dt['g'].sum()
  g01 = sum(dt.apply (lambda row: label_g (row),axis=1))
  print(g0)

  # dt['g'] = dt.apply(lambda row: if row.status == 1 row.a + row.b, axis=1)
  # dt['g'] = dt.apply(lambda row: row.a + row.b, axis=1)
  # dt['g'] = dt.drop('g')
  # return g

def get_h(dt):
  sum =0
  n_row, n_col = dt.shape
  h = 0
  for i in range(n_row):
    if dt['status'].iloc[i] == 1:
      h  += total_weight *2
    else:
      h += 2
  # print(h)
  return h


def get_w(g,h):
  w = -g/(h + lambda_1)
  return w

def get_obj1(g1,h1,g2,h2):
  obj_1 = -0.5 * ( g1*g1/(h1 + lambda_1) + g2*g2/(h2 + lambda_1) )
  return obj_1

def get_obj2(g1,h1,g2,h2):
  obj_1 = -0.5 * ( g1*g1*g1*g1/(h1 + lambda_1) + g2*g2*g2*g2/(h2 + lambda_1) )
  return obj_1


first_step(training_set)
print(1)
#print(training_set)