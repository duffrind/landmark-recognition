# read train.csv -> find top 10 by row count
import pandas as pd

train = pd.read_csv('train.csv')

train.groupby('landmark_id').size().sort_values(ascending=False)[:10]

'''
landmark_id
9633    50337
6051    50148
6599    23415
9779    18471
2061    13271
5554    11147
6651     9508
6696     9222
5376     9216
2743     8997
dtype: int64
'''

train_ids = train.groupby('landmark_id').size()

#ids = train_ids.sort_values(ascending=False)[:10].index.tolist()

ids = train_ids.sort_values(ascending=False)[:2].index.tolist()

#new_train = train[train.landmark_id.isin(ids)]

#new_train.to_csv('new_train.csv', index=False)

ids = [8997, 9216]

newest_train = train[train.landmark_id.isin(ids)]

newest_train.to_csv('newest_train.csv', index=False)

#for id_num in new_train.id:
    #

#training_subset = new_train.random_sample(.7)
# convert each image into a matrix
# training_subset IS NOW an array of matrices
