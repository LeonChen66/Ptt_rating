#-*-coding: utf-8-*-
"""
Created on 2018/3/25 下午 11:04 

@author: Leon
"""
# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

norm_size = 64


def predict(model_name, img_name):
    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model(model_name)

    # load the image
    image = cv2.imread(img_name)
    orig = image.copy()

    # pre-process the image for classification
    image = cv2.resize(image, (norm_size, norm_size))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # import lables
    encoder = LabelEncoder()
    encoder.classes_ = np.load('classes.npy')
    # classify the input image
    result = model.predict(image)[0]
    # print (result.shape)
    print(result)
    proba = np.max(result)
    # inverse label
    label = encoder.inverse_transform(np.where(result == proba)[0])
    # print(label[])
    label = "{}: {:.2f}%".format(label, proba * 100)
    print(label)

    if True:
        # draw the label on the image
        output = imutils.resize(orig, width=400)
        cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        # show the output image
        cv2.imshow("Output", output)
        cv2.waitKey(0)


# python predict.py --model traffic_sign.model -i ../2.png -s
if __name__ == '__main__':
    predict('face.model','1.jpg')