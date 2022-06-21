import cv2 as cv
import os
from PIL import Image
from torchvision.transforms import CenterCrop,Grayscale
import numpy as np

# path_vi = '/Users/zona/Downloads/RoadScene/crop_LR_visible/'
# path_ir = '/Users/zona/Downloads/RoadScene/cropinfrared/'
path = 'ir/'
save_path = 'ir_crop/'

path_2 = 'vi/'
save_path_2 = 'vi_crop/'
# save_path_ir = '/Users/zona/Downloads/RoadScene/ir_q/'
# save_path_vi = '/Users/zona/Downloads/RoadScene/vi_q/'

index = 1
for filename in os.listdir(path):
    # img = cv.imread(path+filename)
    filename_2 = 'V'+filename[1:];

    if os.path.exists(save_path) is False:
        os.mkdir(save_path)
    if os.path.exists(save_path_2) is False:
        os.mkdir(save_path_2)
    img = Image.open(path+filename)
    img_2 = Image.open(path_2+filename_2)


    h = img.size[0]
    w = img.size[1]

    if(h%16!=0):
        h=h-h%16
    if(w%16!=0):
        w=w-w%16
    crop_obj = CenterCrop((w, h))  # 生成一个CenterCrop类的对象,用来将图片从中心裁剪成224*224
    gray_obj = Grayscale(num_output_channels=1)

    img = crop_obj(img)
    img = gray_obj(img)
    img_2 = crop_obj(img_2)
    img_2 = gray_obj(img_2)

    new_name = str(index)+'.bmp'
    index +=1
    img.save(save_path+new_name)
    img_2.save(save_path_2+new_name)
    # img_ir.save(save_path_ir+new_name)
    # img_vi.save(save_path_vi+new_name)





