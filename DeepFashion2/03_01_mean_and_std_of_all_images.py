# train 폴더에 포함된 모든 이미지들에 대해 픽셀의 평균, 표준편차 계산하기
'''
    - input
        - train용 cropped 이미지 파일 (Dataset/train/train/cropped_image)

    - output
        - 평균 및 표준편차: all_image_mean_std.pickle (Dataset/pickle)
             - Mean RGB = mean_std_list[0] = (0.5588169, 0.5191573, 0.51253057)
             - Std RGB  = mean_std_list[1] = (0.24055606, 0.24257304, 0.22783071)
'''
import os
from tqdm import tqdm
import pickle
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# pickle 파일 폴더
pickle_file_folder = "./Dataset/pickle/"

# Train cropped 이미지 폴더 지정
train_cropped_image_folder = "./Dataset/train/train/cropped_image/"

# train cropped image folder에서 파일을 읽어서 파일 목록 생성
train_cropped_file_list = os.listdir(train_cropped_image_folder)


class CustomImageDataset(Dataset):

    def __init__(self, data_file, image_folder, transform=None):
        self.data_file = data_file
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, index):
        img_file = self.data_file[index]
        image = Image.open(os.path.join(self.image_folder, img_file))

        if self.transform:
            image = self.transform(image)

        return image


# 이미지 데이터 파일, 이미지 폴더 지정
data_file = train_cropped_file_list  # train folder에 있는 전체 이미지를 대상으로 함
image_folder = train_cropped_image_folder

transform_data = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
all_train_image = CustomImageDataset(data_file=data_file, image_folder=image_folder, transform=transform_data)

# 이미지 픽셀의 평균, 표준편차 계산
print(' => Calculating image pixels mean and standard deviation... ')

meanRGB = [np.mean(x.numpy(), axis=(1, 2)) for x in tqdm(all_train_image)]
stdRGB = [np.std(x.numpy(), axis=(1, 2)) for x in tqdm(all_train_image)]

meanR = np.mean([m[0] for m in meanRGB])
meanG = np.mean([m[1] for m in meanRGB])
meanB = np.mean([m[2] for m in meanRGB])

stdR = np.mean([s[0] for s in stdRGB])
stdG = np.mean([s[1] for s in stdRGB])
stdB = np.mean([s[2] for s in stdRGB])

print(' -> Mean RGB =', meanR, meanG, meanB)
print(' -> Standard Deviation RGB =', stdR, stdG, stdB)

mean_std_list = [(meanR, meanG, meanB), (stdR, stdG, stdB)]
# Save pickle
with open(pickle_file_folder + "all_image_mean_std.pickle", "wb") as fw:  # 전체 이미지들의 픽셀 평균, 표준편차
    pickle.dump(mean_std_list, fw)

# Load pickle
with open(pickle_file_folder + "all_image_mean_std.pickle", "rb") as fr:
    mean_std_list = pickle.load(fr)

print("== 확인 ==")
print(' - Mean RGB =', mean_std_list[0])
print(' - Standard Deviation RGB =', mean_std_list[1])
