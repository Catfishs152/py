import numpy as np
import cv2 as cv

path = r'C:\Users\11791\Desktop\Tohru.jpg'    #图片存储路径，注意不能有中文
def Gray_img(img):
    # img_gray = cv.cvtColor(img_2, cv.COLOR_RGB2GRAY)
    print(img.shape)
    (h, w, c) = img.shape
    img_b = img[:, :, 0]
    img_g = img[:, :, 1]
    img_r = img[:, :, 2]
    img_gray = img_r * 0.299 + img_g * 0.587 + img_b * 0.114
    img_gray = img_gray.astype(np.uint8)  # (1)
    cv.imshow('', img_gray)
    cv.waitKey(0)
    return img_gray

img = cv.imread(path)
img_gray = Gray_img(img)
