import pandas as pd

train = pd.read_csv('train.csv')

#train_ids = train.groupby('landmark_id').size()

#train_ids = train.groupby('landmark_id').size().index[train.groupby('landmark_id').size() > 5000].tolist()

train_ids = train.groupby('landmark_id').size().sort_values(ascending=False)[:10].index.tolist()

new_train = train[train.landmark_id.isin(train_ids)]

grouped = new_train.groupby('landmark_id')

new_train = grouped.apply(lambda x: x.sample(n=5000))

ids = dict()
i = 0

def get_id(cur_id):
    global i
    global ids
    if ids.get(cur_id, -1) == -1:
        ids[cur_id] = i
        i += 1
    return ids[cur_id]

new_train.landmark_id = new_train.landmark_id.apply(lambda id: get_id(id))

new_train.to_csv('new_train.csv', index=False)

print(len(train_ids))

# write list(new_train.landmark_id.unique) to file, one entry per row
