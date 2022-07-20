# Train data set, validation data set 생성
#     - train용 positive, negative pair를 가지고 train data set 생성
#     - validation용 pair로 validation data set 생성
'''
    - input
        - train positive and negative pairs:      train_positive_pairs.pickle, train_negative_pairs.pickle (Dataset/pickle)
        - validation positive and negative pairs: valid_positive_pairs.pickle, valid_negative_pairs.pickle (Dataset/pickle)

    - output
        - train data set:       train_dataset.pickle (Dataset/pickle)
        - validation data set:  valid_dataset.pickle (Dataset/pickle)

    - remark
        - 각 pair에 해당하는 이미지 파일 이름을 만들어서 저장한다
        - dataset = positive image file pairs + negative image file pairs
'''
import os
import numpy as np
from tqdm import tqdm
import pickle
import random

# pickle 파일 폴더
pickle_file_folder = "./Dataset/pickle/"

####################################################################################################
# Train pair로 data set 만들기
####################################################################################################
# Load pickle
with open(pickle_file_folder + "train_positive_pairs.pickle", "rb") as fr:
    train_positive_pairs = pickle.load(fr)
with open(pickle_file_folder + "train_negative_pairs.pickle", "rb") as fr:
    train_negative_pairs = pickle.load(fr)

# train dataset 만들기
train_positive_dataset = []
for p1, p2, label in tqdm(train_positive_pairs):
    temp1 = p1['pair_id'] + '_' + p1['style'] + '_' + p1['source'] + '_' + p1['filename'] + '_' + p1['item_no'] \
            + '_' + p1['category_id'] + '.jpg'
    temp2 = p2['pair_id'] + '_' + p2['style'] + '_' + p2['source'] + '_' + p2['filename'] + '_' + p2['item_no'] \
            + '_' + p2['category_id'] + '.jpg'
    train_positive_dataset.append((temp1, temp2, label))

train_negative_dataset = []
for p1, p2, label in tqdm(train_negative_pairs):
    temp1 = p1['pair_id'] + '_' + p1['style'] + '_' + p1['source'] + '_' + p1['filename'] + '_' + p1['item_no'] \
            + '_' + p1['category_id'] + '.jpg'
    temp2 = p2['pair_id'] + '_' + p2['style'] + '_' + p2['source'] + '_' + p2['filename'] + '_' + p2['item_no'] \
            + '_' + p2['category_id'] + '.jpg'
    train_negative_dataset.append((temp1, temp2, label))

train_dataset = train_positive_dataset + train_negative_dataset

# train dataset 저장 폴더 지정
train_dataset_folder = "Dataset/train/train/train_dataset/"
if not os.path.isdir(train_dataset_folder):
    os.mkdir(train_dataset_folder)

# Save pickle
with open(train_dataset_folder + "train_dataset.pickle", "wb") as fw:  # train data set 파일
    pickle.dump(train_dataset, fw)

print("Save train dataset... finished!")
print("  -> # of train dataset:", len(train_dataset))

####################################################################################################
# Validation pair로 data set 만들기
####################################################################################################
# Load pickle
with open (pickle_file_folder + "valid_positive_pairs.pickle", "rb") as fr:
    valid_positive_pairs = pickle.load(fr)
with open (pickle_file_folder + "valid_negative_pairs.pickle", "rb") as fr:
    valid_negative_pairs = pickle.load(fr)

# validation dataset 만들기
valid_positive_dataset = []
for p1, p2, label in tqdm(valid_positive_pairs):
    temp1 = p1['pair_id'] + '_' + p1['style'] + '_' + p1['source'] + '_' + p1['filename'] + '_' + p1['item_no'] \
    + '_' + p1['category_id'] + '.jpg'
    temp2 = p2['pair_id'] + '_' + p2['style'] + '_' + p2['source'] + '_' + p2['filename'] + '_' + p2['item_no'] \
    + '_' + p2['category_id'] + '.jpg'
    valid_positive_dataset.append((temp1, temp2, label))

valid_negative_dataset = []
for p1, p2, label in tqdm(valid_negative_pairs):
    temp1 = p1['pair_id'] + '_' + p1['style'] + '_' + p1['source'] + '_' + p1['filename'] + '_' + p1['item_no'] \
    + '_' + p1['category_id'] + '.jpg'
    temp2 = p2['pair_id'] + '_' + p2['style'] + '_' + p2['source'] + '_' + p2['filename'] + '_' + p2['item_no'] \
    + '_' + p2['category_id'] + '.jpg'
    valid_negative_dataset.append((temp1, temp2, label))

valid_dataset = valid_positive_dataset + valid_negative_dataset

# valid dataset 저장 폴더 지정
valid_dataset_folder = "Dataset/validation/validation/validation_dataset/"
if not os.path.isdir(valid_dataset_folder):
    os.mkdir(valid_dataset_folder)

# Save pickle
with open (valid_dataset_folder + "valid_dataset.pickle", "wb") as fw:  # validation data set 파일
    pickle.dump(valid_dataset, fw)

print("Save valid dataset... finished!")
print("  -> # of valid dataset:", len(valid_dataset))
