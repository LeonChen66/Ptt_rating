#-*-coding: utf-8-*-
"""
Created on 2018/3/25 下午 09:52 

@author: Leon
"""

# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import StratifiedKFold
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
import sys
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
import resnest
sys.path.append('..')
from lenet import LeNet
from facerank import make_network

# initialize the number of epochs to train for, initial learning rate,
# and batch size
EPOCHS = 20
INIT_LR = 1e-3
BS = 10
CLASS_NUM = 5
norm_size = 64
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
csv_logger = CSVLogger('resnet18_cifar10.csv')


def load_data(path):
    print("[INFO] loading images...")
    data = []
    labels = []
    # grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images(path)))
    random.seed(42)
    random.shuffle(imagePaths)
    # loop over the input images
    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        try:
            image = cv2.resize(image, (norm_size, norm_size))
            image = img_to_array(image)
            data.append(image)

        # extract the class label from the image path and update the
        # labels list
            label = imagePath.split(os.path.sep)[-2]
            labels.append(label)
        except:
            pass

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float32") / 255.0
    labels = np.array(labels)
    label_encoder = LabelEncoder()    # string label to integer
    integer_encoded = label_encoder.fit_transform(labels)
    # convert the labels from integers to vectors
    labels = to_categorical(integer_encoded, num_classes=5)
    np.save('classes.npy',label_encoder.classes_)
    return data, labels


def train(aug, trainX, trainY, testX, testY, args):
    # initialize the model
    print("[INFO] compiling model...")
    # base_model = VGG19(include_top=False ,weights=None,input_shape=(128,128,3),classes=5)
    # Lenet-5 model and Inception V3
    # create the base pre-trained model
    # base_model = InceptionV3(weights='imagenet', include_top=False,input_shape=(150,150,3),classes=5)

    # add a global spatial average pooling layer
    # x = base_model.output
    # x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    # x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    # predictions = Dense(5, activation='softmax')(x)
    # model = Model(inputs=base_model.input, outputs=predictions)
    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=["accuracy"])

    model = LeNet.build(width=norm_size, height=norm_size, depth=3, classes=CLASS_NUM)
    # opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer='adam',
                  metrics=["accuracy"])
    #
    # train the network
    print("[INFO] training network...")
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
                            validation_data=(testX, testY), steps_per_epoch=len(trainX) // (BS),
                            epochs=EPOCHS, verbose=1)

    # Resent Model
    # model = resnest.ResnetBuilder.build_resnet_34((3, norm_size, norm_size), CLASS_NUM)
    # model.compile(loss='categorical_crossentropy',
    #               optimizer='adam',
    #               metrics=['accuracy'])
    # print("[INFO] training network...")
    # H = model.fit(trainX, trainY,
    #           batch_size=BS,
    #           nb_epoch=EPOCHS,
    #           validation_data=(testX, testY),
    #           shuffle=True,
    #           class_weight='auto',
    #           callbacks=[lr_reducer, early_stopper, csv_logger])
    # model = make_network()
    # model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    # H = model.fit(trainX, trainY,
    #               batch_size=BS,
    #               validation_data=(testX,testY),
    #               epochs=EPOCHS,
    #               verbose=1,
    #               shuffle=True
    #               )

    print("[INFO] serializing network...")
    # save model
    model.save(args)
    print(model.summary(),'\n')
    scores = model.evaluate(testX,testY)
    print(scores)
    print("model accuracy: {:.2f}".format(scores[1]))


    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), np.array(H.history["loss"]), label="train_loss")
    plt.plot(np.arange(0, N), np.array(H.history["val_loss"]), label="val_loss")
    plt.plot(np.arange(0, N), np.array(H.history["acc"]), label="train_acc")
    plt.plot(np.arange(0, N), np.array(H.history["val_acc"]), label="val_acc")
    plt.title("Training Loss and Accuracy on traffic-sign classifier")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("test.png")
    plt.show()
    return model


def main():
    image,label = load_data('face')
    trainX, testX, trainY, testY = train_test_split(
    image, label, test_size = 0.33, random_state = 42)
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, fill_mode="nearest")

    model = train(aug,trainX,trainY,testX,testY,'face.model')
    print(model.summary(),'\n')
    scores = model.evaluate(testX,testY)
    print("model accuracy: {:.2f}".format(scores[1]))


if __name__=='__main__':
    main()