import os
import pandas as pd
from shutil import copyfile
#import numpy as np

files = pd.read_csv('train.csv')

#files['random_number'] = np.random.randn(files.shape[0])

#train = files[files['random_number'] <= 0.7]
#test = files[files['random_number'] > 0.7]

for _, row in files.iterrows():
    if os.path.isfile('train/' + str(row.id) + '.jpg'):
        if not os.path.exists('all_photos/' + str(row.landmark_id)):
            os.makedirs('all_photos/' + str(row.landmark_id))
        _ = copyfile('train/' + str(row.id) + '.jpg', 'all_photos/' + str(row.landmark_id) + '/' + str(row.id) + '.jpg')

#for _, row in test.iterrows():
#    if os.path.isfile('train/' + str(row.id) + '.jpg'):
#        if not os.path.exists('validation/' + str(row.landmark_id)):
#            os.makedirs('validation/' + str(row.landmark_id))
#        _ = copyfile('train/' + str(row.id) + '.jpg', 'validation/' + str(row.landmark_id) + '/' + str(row.id) + '.jpg')
