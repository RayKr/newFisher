# -*- coding: -utf-8 -*-
# Python版本：Python3.5
# 库：numpy，opencv，matplotlib
# 基于离散余弦变换DCT的图像压缩
#  作者：James_Ray_Murpy
# 2018/01/25

import cv2
import numpy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from file import image_dct

img = cv2.imread('3.png', 0)  # 读取图片，
img1 = img.astype('float')  # 将uint8转化为float类型
img_dct = cv2.dct(img1)  # 进行离散余弦变换
img_dct_log = np.log(abs(img_dct))  # 进行log处理
img_recor = cv2.idct(img_dct)  # 进行离散余弦反变换

# 图片压缩，只保留100*100的数据
recor_temp = img_dct[0:26, 0:26]
recor_temp2 = np.zeros(img.shape)
recor_temp2[0:26, 0:26] = recor_temp
# 压缩图片恢复
img_recor1 = cv2.idct(recor_temp2)

# 显示
# plt.subplot(221)
# plt.imshow(img)
# plt.title('original')
#
# plt.subplot(222)
# plt.imshow(img_dct_log)
# plt.title('dct transformed')
#
# plt.subplot(223)
# plt.imshow(img_recor)
# plt.title('idct transformed')
#
# plt.subplot(224)
# plt.imshow(img_recor1)
# plt.title('idct transformed2')
#
# plt.show()


def image_denoise(filename):
    cv_image = cv2.imread(filename)  # 读取图片，
    denoise = cv2.fastNlMeansDenoisingColored(cv_image, None, 15, 15, 7, 21)
    return denoise


# deimg = image_denoise('3.png')
# plt.subplot(222)
# plt.imshow(img)
# plt.show()


img = image_dct('space.jpeg', 500)
img.show()
