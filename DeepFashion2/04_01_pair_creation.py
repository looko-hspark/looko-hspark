# Train data set, validation data set 생성을 위한 pair 생성
#     - train image set으로 train용 positive, negative pair 생성
#     - validation image set으로 validation용 pair 생성
'''
    - input
        - train image set:       X_train_image.pickle (Dataset/pickle)
        - validation image set:  X_valid_image.pickle (Dataset/pickle)

    - output
        - train positive and negative pairs:      train_positive_pairs.pickle, train_negative_pairs.pickle (Dataset/pickle)
        - validation positive and negative pairs: valid_positive_pairs.pickle, valid_negative_pairs.pickle (Dataset/pickle)

    - remark
        - # of negative pairs = # of positive pairs x times (= 3)
        - train image set을 user group과 shop group으로 나눈다
            - user group의 이미지에 대해 shop group에서 같은 것을 찾아 positive pair를 만든다
            - pair가 되지 못한 나머지 이미지들 중에서 랜덤 샘플링해서 negative pair를 만든다 (지정한 숫자 만큼)
        - validation image set에 대해서도 반복한다
'''
import os
from PIL import Image
from tqdm import tqdm
import pickle
import random

# pickle 파일 폴더 지정
pickle_file_folder = "./Dataset/pickle/"

# negative pairs 개수 (= positive pairs 개수 * times) 지정
times = 3

####################################################################################################
# Train image set으로 pair 만들기
####################################################################################################
# Load pickle: train image set
with open(pickle_file_folder + "X_train_image.pickle", "rb") as fr:
    X_train_image = pickle.load(fr)

print('No. of X_train_image:', len(X_train_image))

# X_train_image를 user그룹과 shop그룹으로 나누기
#        => user 그룹에서는 style = '00'인 품목은 pair 구성 대상에서 제외

user_group = []
shop_group = []
for i in X_train_image:
    if i['source'] == 'user':
        if i['style'] != '00':
            user_group.append(i)
        else:
            continue
    elif i['source'] == 'shop':
        shop_group.append(i)
    else:
        print('Warning! Wrong Source...')

print('=> train user_group length:', len(user_group))
print('=> train shop_group length:', len(shop_group))

# Pair 생성: user그룹에서 하나 뽑은 후 해당하는 X_train_image를 가져오고
#            shop그룹의 모든 item에 대해서 해당하는 X_train_image를 가져와서 비교하여
#            category_id, pair_id, style이 모두 같으면 positive_pair

positive_pairs = []
negative_pairs = []
temp_pairs = []
pair_num = 0

for a_user in tqdm(user_group):
    for a_shop in shop_group:
        if a_user['pair_id'] == a_shop['pair_id'] and a_user['style'] == a_shop['style'] and a_user['category_id'] == \
                a_shop['category_id']:
            positive_pairs.append((a_user, a_shop, 1))  # True = 1... label 추가
            pair_num += 1  # positive pair 개수
        else:
            temp_pairs.append(a_shop)  # negative pair 구성을 위한 모집단

    # positive pair 구성이 끝나면 nagative pair 구성
    #      => 지정한 수 (= pair_num * times) 만큼 temp_pairs에서 random sampling 하여 negative pair에 추가

    if pair_num != 0:
        neg_sample = random.sample(temp_pairs, pair_num * times)

        for i in neg_sample:
            negative_pairs.append((a_user, i, 0))  # False = 0... label 추가

    # 다음 user 이미지에 대한 작업을 위해 초기화
    temp_pairs = []
    pair_num = 0
    neg_sample = []

# Save pickle
with open(pickle_file_folder + 'train_positive_pairs.pickle', 'wb') as fw:
    pickle.dump(positive_pairs, fw)
with open(pickle_file_folder + 'train_negative_pairs.pickle', 'wb') as fw:
    pickle.dump(negative_pairs, fw)

print("Save postive and negative pairs, respectively... finished!")
print("  -> # of positive pairs:", len(positive_pairs))
print("  -> # of negative pairs:", len(negative_pairs))

####################################################################################################
# Validation image set으로 pair 만들기
####################################################################################################
# Load pickle: validation image set
with open(pickle_file_folder + "X_valid_image.pickle", "rb") as fr:
    X_valid_image = pickle.load(fr)

print('No. of X_valid_image:', len(X_valid_image))

# X_valid_image를 user그룹과 shop그룹으로 나누기
#        => user 그룹에서는 style = '00'인 품목은 pair 구성 대상에서 제외

user_group = []
shop_group = []
for i in X_valid_image:
    if i['source'] == 'user':
        if i['style'] != '00':
            user_group.append(i)
        else:
            continue
    elif i['source'] == 'shop':
        shop_group.append(i)
    else:
        print('Warning! Wrong Source...')

print('=> valid user_group length:', len(user_group))
print('=> valid shop_group length:', len(shop_group))

# Pair 생성: user그룹에서 하나 뽑은 후 해당하는 X_valid_image를 가져오고
#            shop그룹의 모든 item에 대해서 해당하는 X_valid_image를 가져와서 비교하여
#            category_id, pair_id, style이 모두 같으면 positive_pair

positive_pairs = []
negative_pairs = []
temp_pairs = []
pair_num = 0

for a_user in tqdm(user_group):
    for a_shop in shop_group:
        if a_user['pair_id'] == a_shop['pair_id'] and a_user['style'] == a_shop['style'] and a_user['category_id'] == \
                a_shop['category_id']:
            positive_pairs.append((a_user, a_shop, 1))  # True = 1... label 추가
            pair_num += 1  # positive pair 개수
        else:
            temp_pairs.append(a_shop)  # negative pair 구성을 위한 모집단

    # positive pair 구성이 끝나면 nagative pair 구성
    #      => 지정한 수 (= pair_num * times) 만큼 temp_pairs에서 random sampling 하여 negative pair에 추가

    if pair_num != 0:
        neg_sample = random.sample(temp_pairs, pair_num * times)

        for i in neg_sample:
            negative_pairs.append((a_user, i, 0))  # False = 0... label 추가

    # 다음 user 이미지에 대한 작업을 위해 초기화
    temp_pairs = []
    pair_num = 0
    neg_sample = []

# Save pickle
with open(pickle_file_folder + 'valid_positive_pairs.pickle', 'wb') as fw:
    pickle.dump(positive_pairs, fw)
with open(pickle_file_folder + 'valid_negative_pairs.pickle', 'wb') as fw:
    pickle.dump(negative_pairs, fw)

print("Save postive and negative pairs, respectively... finished!")
print("  -> # of positive pairs:", len(positive_pairs))
print("  -> # of negative pairs:", len(negative_pairs))
