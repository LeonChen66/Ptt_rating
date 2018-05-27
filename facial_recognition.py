#-*-coding: utf-8-*-
"""
Created on 2018/3/24 下午 08:30 

@author: Leon
"""

import cv2
import sys
import os
from tqdm import tqdm

img = "test7.jpg"
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def facial_reg(img):
    name = img
    gray =cv2.imread(img,0)
    faces = face_cascade.detectMultiScale(gray,1.6,6)
    return faces

def save_face(img_name,faces,file_num):
    img = cv2.imread(img_name)
    for index, (x,y,w,h) in enumerate(faces):
        try:
            crop_img = img[y:y + h, x:x + w]
            # print(crop_img)
            # cv2.imshow('img', crop_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            img_name = os.path.split(img_name)[1]
            try:
                os.mkdir('face/'+file_num)
                name = "face/{!s}/{!s}_{!s}".format(file_num, index, img_name)
            except:
                name = "face/{!s}/{!s}_{!s}".format(file_num, index, img_name)

            cv2.imwrite(name, crop_img)
        except:
            pass


def main():
    for root, dirs, files in os.walk("full"):
        for f in tqdm(files):
            img_name = os.path.join(root, f)
            face = facial_reg(img_name)
            file_num = root.split('\\')[1]
            save_face(img_name,face, file_num)
    # facial_reg('0WjhU6D.jpg')


if __name__=="__main__":
    main()
