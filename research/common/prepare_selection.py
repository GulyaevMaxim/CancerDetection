import numpy as np
import pandas
import json
import numpy.random as random

train_images = pandas.read_csv('train_labels.csv')

train_img_names = train_images.values[:, 0]
predicted_labels = train_images.values[:, 1]

valid_id = 0
train_id = 0

df_valid = pandas.DataFrame(columns=['id', 'label'])
df_train = pandas.DataFrame(columns=['id', 'label'])

not_marked = np.where(predicted_labels == 0)
marked = np.where(predicted_labels == 1)

size_zero = len(not_marked[0])
valid_size_zero = int(0.2 * size_zero)

size_one = len(marked[0])
valid_size_one = int(0.2 * size_one)

pr_id = 0

for i in range(valid_size_zero):
    file_name = train_img_names[not_marked[0][pr_id]]
    df_valid.at[valid_id, 'id'] = file_name
    df_valid.at[valid_id, 'label'] = 0
    valid_id += 1
    pr_id += 1

train_size = size_zero - valid_size_zero

for i in range(train_size):
    file_name = train_img_names[not_marked[0][pr_id]]
    df_train.at[train_id, 'id'] = file_name
    df_train.at[train_id, 'label'] = 0
    train_id += 1
    pr_id += 1

print('0 is implemented')
pr_id = 0

for i in range(valid_size_one):
    file_name = train_img_names[marked[0][pr_id]]
    df_valid.at[valid_id, 'id'] = file_name
    df_valid.at[valid_id, 'label'] = 1
    valid_id += 1
    pr_id += 1

train_size = size_one - valid_size_one

for i in range(train_size):
    file_name = train_img_names[marked[0][pr_id]]
    df_train.at[train_id, 'id'] = file_name
    df_train.at[train_id, 'label'] = 1
    train_id += 1
    pr_id += 1

print('1 is implemented')

df_train = df_train.iloc[random.permutation(len(df_train))]
df_valid = df_valid.iloc[random.permutation(len(df_valid))]

df_train.to_csv('my_train.csv')
df_valid.to_csv('my_valid.csv')
