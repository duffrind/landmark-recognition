import pandas as pd

train = pd.read_csv('train.csv')

train_ids = train.groupby('landmark_id').size().index[train.groupby('landmark_id').size() > 50].tolist()

new_train = train[train.landmark_id.isin(train_ids)]

new_train.to_csv('new_train.csv', index=False)
