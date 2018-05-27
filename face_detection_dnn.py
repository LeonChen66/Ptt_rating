#-*-coding: utf-8-*-
"""
Created on 2018/3/27 下午 01:09 

@author: Leon
"""
import cv2
from imutils import paths
import numpy as np
from tqdm import tqdm
import os

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", 'res10_300x300_ssd_iter_140000.caffemodel')

def detect_save(img_name,file_num):
    try:
        image = cv2.imread(img_name)
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the detections and
        # predictions
        # print("[INFO] computing object detections...")
        net.setInput(blob)
        detections = net.forward()
        index = 0
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > 0.8:
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                # draw the bounding box of the face along with the associated
                # probability
                # text = "{:.2f}%".format(confidence * 100)
                # y = startY - 10 if startY - 10 > 10 else startY + 10
                # cv2.rectangle(image, (startX, startY), (endX, endY),
                #               (0, 0, 255), 2)
                # cv2.putText(image, text, (startX, y),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                img_name = os.path.split(img_name)[1]
                if startY>=0 and startX>=0 and endY<h and endX<w:
                    try:
                        os.mkdir('face/' + file_num)
                        name = "face/{!s}/{!s}_{!s}".format(file_num, index, img_name)
                    except:
                        name = "face/{!s}/{!s}_{!s}".format(file_num, index, img_name)

                    try:
                        cv2.imwrite(name,image[startY:endY, startX:endX])
                        index+=1
                    except:
                        pass
    except:
        print(img_name)
        # show the output image
    # cv2.imshow("Output", image)
    # cv2.waitKey(0)

def main():
    for root, dirs, files in os.walk("full"):
        for f in tqdm(files):
            img_name = os.path.join(root, f)
            file_num = root.split('\\')[1]
            detect_save(img_name,file_num)

if __name__=="__main__":
    main()