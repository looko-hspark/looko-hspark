import os
import json
from PIL import Image
import glob
from tqdm import tqdm

# Json, 이미지 폴더 지정
json_folder = "Dataset/train/train/annos/"
image_folder = "Dataset/train/train/image/"

# 생성되는 Cropped 이미지 저장 폴더 지정
cropped_image_folder = "Dataset/train/train/cropped_image/"
if not os.path.isdir(cropped_image_folder):
    os.mkdir(cropped_image_folder)

# 각각의 Json파일에서 데이터를 읽고
# 이를 사용하여 cropped 이미지를 만들어야 하므로... Json 폴더 내의 파일 목록 생성
json_file_list = os.listdir(json_folder)

# 하나의 Json 파일에 대하여...
for json_file in tqdm(json_file_list):
    json_filename = json_folder + json_file

    # Json 파일 열기
    with open(json_filename, "r") as f:
        js = json.load(f)
    # Source(=user or shop)와 pair_id 저장
    source = js['source']
    pair_id = js['pair_id']

    for i in js.keys():

        if i == 'source' or i == 'pair_id':
            continue

        # item들에 대하여 style, bounding_box, category_id 저장
        else:
            style = js[i]['style']
            bounding_box = js[i]['bounding_box']
            category_id = js[i]['category_id']

        # 이미지 crop 및 cropped 이미지 저장
        # cropped 이미지 파일 이름 구성: pair_id + style + source + filename 앞부분 + item# + category_id + ".jpg"

        #  json 파일 이름(확장자 제외)과 같은 이름의 이미지 파일 열기
        image_filename = image_folder + json_file.rstrip('.json') + '.jpg'
        image1 = Image.open(image_filename)

        # 이미지 crop
        if bounding_box == [0, 0, 0, 0]:  # 바운딩 박스 오류가 있다고 함...
            continue

        else:
            cropped_image = image1.crop(bounding_box)

            # Cropped 이미지 파일 이름 만들기: pair_id + style + source + finename + item# + category_id + ".jpg"
            crop_filename = str('{0:06d}'.format(pair_id)) + '_' \
                            + str('{0:02d}'.format(style)) + '_' \
                            + source + '_' \
                            + json_file.rstrip('.json') + '_' \
                            + str(i) + '_' \
                            + str('{0:02d}'.format(category_id)) + '.jpg'

            # Cropped 이미지 저장하기
            cropped_image.save(cropped_image_folder + crop_filename)

print("Cropped images save... finished!")
