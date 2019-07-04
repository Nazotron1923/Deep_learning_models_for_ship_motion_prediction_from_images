"""
joint several pictures together
"""
import os
from PIL import Image
from tqdm import tqdm
import cv2

width = 486
height = 270
TARGET_WIDTH = 3 * width
path1 = "/home/theo/Study/Cours_en_france/Pre/Code/horizon-detection/Pre/3dmodel/rendercnnone"
path2 = "/home/theo/Study/Cours_en_france/Pre/Code/horizon-detection/Pre/3dmodel/rendercnntwo"
path3 = "/home/theo/Study/Cours_en_france/Pre/Code/horizon-detection/Pre/3dmodel/rendercnnlstm"
path = [path1, path2, path3]
res_path = "/home/theo/Study/Cours_en_france/Pre/Code/horizon-detection/Pre/3dmodel"
pic_item = 3

def read_filename(dir):
    image_list = []
    for root, dirs, files in os.walk(dir):
        for f in files :
            if f.endswith('png'):
                image_list.append(f)
    image_list.sort(key=lambda name: int(name.split('.')[0]))
    return image_list
        # print(images)
        # print(len(images))
image_list1 = read_filename(path1)
image_list2 = read_filename(path2)
image_list3 = read_filename(path3)
image_list = [image_list1, image_list2, image_list3]
text = ['CNN_ONE', 'CNN_TWO', 'CNN_LSTM']
file_num = min(len(image_list1),len(image_list2),len(image_list3))
img_idx = 0
for i in tqdm(range(file_num)):
    imagefile = []
    for j in range(pic_item):
        # tmp = Image.open(path[j]+'/'+image_list[j][img_idx])
        tmp = cv2.imread(path[j]+'/'+image_list[j][img_idx])
        cv2.putText(tmp, text[j], (0, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255))
        tmp[...,[0,2]] = tmp[...,[2,0]] # RGB-> BGR
        tmp = Image.fromarray(tmp)
        imagefile.append(tmp)
    target = Image.new('RGB', (TARGET_WIDTH, height))
    left = 0
    right = width
    for image in imagefile:
        target.paste(image, (left, 0, right, height))
        left += width
        right += width
        quality_value = 100
        target.save(res_path+'/result/'+os.path.splitext(image_list[0][img_idx])[0]+'.png', quality = quality_value)
    imagefile = []
    img_idx += 1
