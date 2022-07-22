# DeepFashion2 Feature Network

# Train, validation data에 대한 feature descriptor (vector) 생성
#     - ResNet152 pretrained 모델 이용
'''
    - Input
        - train data set:       train_dataset.pickle (Dataset/train/train/train_dataset/)
        - validation data set:  valid_dataset.pickle (Dataset/validation/validation/validation_dataset/)

    - Output
        - train feature vectors: train_feature_vectors.pickle (Dataset/train/train/train_dataset/)
        - validation feature vectors: validation_feature_vectors.pickle (Dataset/validation/validation/validation_dataset/)

    - remark
        - ResNet152 pretrained 모델 이용
'''
#############################################################
# 0. 필요 라이브러리 호출
#############################################################
import os

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn

from torch import nn

from torchvision import datasets, models, transforms
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import pickle
import random

# CPU, GPU 장치 확인
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#############################################################
# 0.1. 이미지 전처리 정의
#############################################################

class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'valid': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, img, phase):
        return self.data_transform[phase](img)

#############################################################
# 1. Train, Validation 데이터셋 불러오기
#############################################################

# === Train 데이터셋 ===
# train 이미지 폴더 지정
train_image_folder = "./Dataset/train/train/cropped_image/"
# train dataset 폴더 지정
train_data_folder = "./Dataset/train/train/train_dataset/"
# train dataset 읽기... Load pickle
with open (train_data_folder + "train_dataset.pickle", "rb") as fr:
    train_data = pickle.load(fr)
print(' -> # of train dataset:', len(train_data))

# === Validation 데이터셋 ===
# validation 이미지 폴더 지정
valid_image_folder = "./Dataset/train/train/cropped_image/"  # >>>>>>>>>>>>>>>>>>>>> 주의할 것 <<<<<<<<<<<<<<<<<<<<<<
# validation dataset 폴더 지정
valid_data_folder = "./Dataset/validation/validation/validation_dataset/"
# validation dataset 읽기... Load pickle
with open (valid_data_folder + "valid_dataset.pickle", "rb") as fr:
    valid_data = pickle.load(fr)
print(' -> # of validation dataset:', len(valid_data))

#############################################################
# 2. Train, Validation 데이터셋 이미지 확인
#############################################################

# === Train 데이터 확인 ===
print('-> train_data[0]:', train_data[:3])  #  파일 이름 목록 파일
# 이미지 디스플레이
img = Image.open(os.path.join(train_image_folder, train_data[0][0])) # 이미지 파일
plt.imshow(img, cmap="gray")
plt.show()
print('   train_data[0][0], size:', ToTensor()(img).shape, end='\n\n\n')
# img = ToTensor()(img).unsqueeze(0)
# print(img.shape)

# === Validation 데이터 확인 ===
print('-> valid_data[0]:', valid_data[0])  #  파일 이름 목록 파일
# 이미지 디스플레이
img = Image.open(os.path.join(valid_image_folder, valid_data[0][0])) # 이미지 파일
plt.imshow(img, cmap="gray")
plt.show()
print('   valid_data[0][0], size:', ToTensor()(img).shape)

#############################################################
# 3. Train, Validation 이미지셋 만들기
#############################################################

# === Train 이미지셋 ===
train_images_filepaths = [[os.path.join(train_image_folder, f1),
                           os.path.join(train_image_folder, f2), label] for f1, f2, label in tqdm(train_data)]
random.seed(42)
random.shuffle(train_images_filepaths)
print(' -> # of train images:', len(train_images_filepaths))

# === Validation 이미지셋 ===
valid_images_filepaths = [[os.path.join(valid_image_folder, f1),
                           os.path.join(valid_image_folder, f2), label] for f1, f2, label in tqdm(valid_data)]
random.seed(42)
random.shuffle(valid_images_filepaths)
print(' -> # of valid images:', len(valid_images_filepaths))

#############################################################
# 4. 변수 값 정의
#############################################################

size = 224  # ResNet에 입력되는 이미지 크기: 224 x 224

# 전체 이미지 평균, 표준편차 값 읽어오기 (03_01_mean_and_std_of_all_images에서 계산)
# pickle 파일 폴더 지정
pickle_file_folder = "./Dataset/pickle/"
# Load pickle
with open (pickle_file_folder + "all_image_mean_std.pickle", "rb") as fr:
    mean_std_list = pickle.load(fr)

mean = mean_std_list[0]
std = mean_std_list[1]
print(' - Mean RGB =', mean)
print(' - Std. RGB =', std)

#############################################################
# 5. ResNet152 pretrained 모델 정의
#############################################################

resnet152 = models.resnet152(pretrained=True)
modules = list(resnet152.children())[:-1]
resnet152 = nn.Sequential(*modules).to(device)

for p in resnet152.parameters():
    p.requires_grad = False

#############################################################
# 6. ResNet152 pretrained 모델을 사용한 feature descriptor 생성
#############################################################

# === Train 데이터셋 ===

train_feature_vectors = []

with torch.no_grad():
    for train_path_1, train_path_2, label in tqdm(train_images_filepaths):
        img1 = Image.open(train_path_1)
        transform = ImageTransform(size, mean, std)
        img1 = transform(img1, phase='train')
        img1 = img1.unsqueeze(0)
        img1 = img1.to(device)
        resnet152.eval()
        output1 = resnet152(img1)
        output1 = output1.squeeze()
        output1 = output1.tolist()

        img2 = Image.open(train_path_2)
        # transform = ImageTransform(size, mean, std)
        img2 = transform(img2, phase='train')
        img2 = img2.unsqueeze(0)
        img2 = img2.to(device)
        # resnet152.eval()
        output2 = resnet152(img2)
        output2 = output2.squeeze()
        output2 = output2.tolist()

        # output = torch.cat([output1, output2])
        # print(output.size())
        output = output1 + output2
        train_feature_vectors.append((output, label))

print('▶ # of train feature vectors:', len(train_feature_vectors))
print('   feature vector size:', len(train_feature_vectors[0][0]))

# train feature descriptor 저장 폴더 지정
train_dataset_folder = "Dataset/train/train/train_dataset/"
if not os.path.isdir(train_dataset_folder):
    os.mkdir(train_dataset_folder)
# Save pickle
with open(train_dataset_folder + "train_feature_vectors.pickle", "wb") as fw:  # train_feature_vectors 파일
    pickle.dump(train_feature_vectors, fw)
print("Save train feature vectors... finished!")
print("  -> # of train feature vectors dataset:", len(train_feature_vectors))

# === Validation 데이터셋 ===

valid_feature_vectors = []

with torch.no_grad():
    for valid_path_1, valid_path_2, label in tqdm(valid_images_filepaths):
        img1 = Image.open(valid_path_1)
        transform = ImageTransform(size, mean, std)
        img1 = transform(img1, phase='valid')
        img1 = img1.unsqueeze(0)
        img1 = img1.to(device)
        resnet152.eval()
        output1 = resnet152(img1)
        output1 = output1.squeeze()
        output1 = output1.tolist()

        img2 = Image.open(valid_path_2)
        # transform = ImageTransform(size, mean, std)
        img2 = transform(img2, phase='valid')
        img2 = img2.unsqueeze(0)
        img2 = img2.to(device)
        # resnet152.eval()
        output2 = resnet152(img2)
        output2 = output2.squeeze()
        output2 = output2.tolist()

        # output = torch.cat([output1, output2])
        # print(output.size())
        output = output1 + output2
        valid_feature_vectors.append((output, label))

print('▶ # of valid feature vectors:', len(valid_feature_vectors))
print('   feature vector size:', len(valid_feature_vectors[0][0]))

# valid feature descriptor 저장 폴더 지정
valid_dataset_folder = "Dataset/validation/validation/validation_dataset/"
if not os.path.isdir(valid_dataset_folder):
    os.mkdir(valid_dataset_folder)
# Save pickle
with open(valid_dataset_folder + "valid_feature_vectors.pickle", "wb") as fw:  # valid_feature_vectors 파일
    pickle.dump(valid_feature_vectors, fw)
print("Save valid feature vectors... finished!")
print("  -> # of valid feature vectors dataset:", len(valid_feature_vectors))
