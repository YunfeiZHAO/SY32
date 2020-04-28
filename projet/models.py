import os
import time
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from skimage import data, color, feature
from skimage.io import imread
from skimage.io import imsave
from skimage.io import imshow
from skimage import transform
from skimage.util import img_as_float
from skimage.transform import rescale

from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score


'''size of window prodefinded by the mean value de labeled boxs'''
HEIGHT = 180
WIDTH = 120

class Box:
    '''
        Definition of Box and Image object in this project
        box: a region on a image
    '''
    def __init__(self, i, j, h, l, s=0):
        self.true_positive = False
        self.x = j
        self.y = i
        self.height = h
        self.width = l
        self.score = s

    def __lt__(self, box):
        return self.score < box.score

    def intersect_over_union(self, box):
        '''
        Calculates the overlap area between two box
        Parameters
        ----------
        k: image number
        i, j: coordinate(row colonne) of the upper left corner of the box
        h, l: size (height, width) of the box
        s: detection score
        '''
        width_intersect = min(self.x + self.width, box.x + box.width) - max(self.x, box.x)
        height_intersect = min(self.y + self.height, box.y + box.height) - max(self.y, box.y)
        # no overleap
        if width_intersect <= 0 or height_intersect <= 0:
            return 0
        else:
            I = width_intersect * height_intersect
            # Union = Total area - Intersection
            U = self.height * self.width + box.height * box.width - I
            return I / U
            # test if a box is true positive

    def test_true_positive(self, label_box, rate=0.5):
        if self.intersect_over_union(label_box) > rate:
            self.true_positive = True
            return self.true_positive

class Image:
    '''
    Image
    attributes
    ----------
    image: a image of nparray type
    number: image index
    face: box which is labeled as a face
    detected: box which is detected by our model
    '''

    def __init__(self, number, image):
        self.image = image
        self.number = number
        self.face = []
        self.detected = []

    def add_face(self, box):
        self.face.append(box)

    def add_detected(self, box):
        self.detected.append(box)

    # remove duplicate and keep only the one with the best score
    def sort_boxs_by_score(self):
        self.detected.sort(key=lambda box: box.score, reverse=True)

    def remove_duplicates_detected_boxs(self, rate=0.01):
        self.sort_boxs_by_score()
        for box_1 in self.detected:
            for box_2 in self.detected[self.detected.index(box_1) + 1:]:
                if box_1.intersect_over_union(box_2) > rate:
                    self.detected.remove(box_2)

    def show_image_face(self):
        '''show images with boxs on it'''
        fig, ax = plt.subplots()
        ax.imshow(self.image, cmap='gray')
        faces = self.face
        detected = self.detected
        for box in faces:
            ax.add_patch(plt.Rectangle((box.x, box.y), box.width, box.height,
                                       edgecolor='red', alpha=1, lw=2, facecolor='none'))
        for box in detected:
            ax.add_patch(plt.Rectangle((box.x, box.y), box.width, box.height,
                                       edgecolor='blue', alpha=1, lw=2, facecolor='none'))

    def save_image_to_folder(self, path):
        ''' Save an array of images to file
            Args
                path: a path to a specific folder
        '''
        fig, ax = plt.subplots()
        ax.imshow(self.image, cmap='gray')
        faces = self.face
        detected = self.detected
        for box in faces:
            ax.add_patch(plt.Rectangle((box.x, box.y), box.width, box.height,
                                       edgecolor='red', alpha=1, lw=2, facecolor='none'))
        for box in detected:
            ax.add_patch(plt.Rectangle((box.x, box.y), box.width, box.height,
                                       edgecolor='blue', alpha=1, lw=2, facecolor='none'))
        fig.savefig(f'{path}/{self.number}.jpg')
        print(f'images {self.number} saved to: {path}')

class Model:
    def load_combine_pos_neg_data(path_pos, path_neg):
        '''Load negative images and positive images
            args: list folfers of positive images and list of folders of negative images
        '''
        X_train = []
        y_train = []
        p = 0
        n = 0
        for folder_path in path_pos:
            for filename in os.listdir(folder_path):
                im = color.rgb2gray(imread(folder_path+'/'+filename))
                a = im.shape
                if a[0] != 180 or a[1] != 120:
                    imshow(im)
                X_train.append(im)
                y_train.append(1)
                p += 1
        print(f'{p} positive samples loaded')
        for folder_path in path_neg:
            for filename in os.listdir(folder_path):
                im = color.rgb2gray(imread(folder_path+'/'+filename))
                a = im.shape
                if a[0] != 180 or a[1] != 120:
                    imshow(im)
                X_train.append(im)
                y_train.append(0)
                n += 1
        print(f'{n} negative samples loaded')
        return X_train, y_train

    def load_pos_neg_from_images(pimages, nimages):
        '''return X_train y_trans given two array of nparray images'''
        X_train = []
        y_train = []
        for im in pimages:
            a = im.shape
            if a[0] != 180 or a[1] != 120:
                imshow(im)
            X_train.append(im)
            y_train.append(1)
        for im in nimages:
            a = im.shape
            if a[0] != 180 or a[1] != 120:
                imshow(im)
            X_train.append(im)
            y_train.append(0)
        return X_train, y_train

    def flatten(self, X_train, y_train):
        return np.array([im.flatten() for im in X_train])

    def hog(X_train):
        return np.array([feature.hog(im, pixels_per_cell=(4, 4)) for im in X_train])

    # use hoge image as out put
    def hog2(images):
        hog = []
        for im in images:
            fd, hog_image = feature.hog(im, visualize=True)
            hog.append(hog_image)
        return np.reshape(np.array(hog), ((-1, 21600)))

    def Naive_Baise(self, X_train, y_train):
        return cross_val_score(GaussianNB(), X_train, y_train)

    def SVM(X_train, y_train):
        grid = GridSearchCV(LinearSVC(class_weight='balanced'), {'C': [1.0, 2.0, 4.0, 8.0]})
        grid.fit(X_train, y_train)
        print(grid.best_score_)
        print(grid.best_estimator_)
        model = grid.best_estimator_
        model.fit(X_train, y_train)
        return model

    def save_model(model, path):
        ''' Save model as a file
            Args:
                model: a clf
        '''
        pickle.dump(model, open(path, 'wb'))

    def load_model(path):
        ''' Load a model from file
            return value:
                A clf
        '''
        return pickle.load(open(path, 'rb'))
