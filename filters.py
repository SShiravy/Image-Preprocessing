import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


class ImageFilter:
    def __init__(self, img_dir='images'):
        self.morphology_imgs = []
        self.blended = []
        self.binary_imgs = []
        self.hls_imgs = []
        self.img_dir = img_dir
        self.images = []

    def read_imgs(self):
        for img in os.listdir(self.img_dir):
            if img[-3:] == 'png':
                self.images.append(cv2.imread(self.img_dir + '/' + img))

    def hls_filter(self,img_list=None):
        img_list = self.images if img_list is None else img_list
        for img in img_list:
            self.hls_imgs.append(cv2.cvtColor(img, cv2.COLOR_RGB2HLS))
        return self.hls_imgs

    def binary_filter(self,img_list=None):
        img_list = self.images if img_list is None else img_list
        for img in img_list:
            _, binary_imgs = cv2.threshold(img, img.max() / 2, img.max(), cv2.THRESH_BINARY)
            self.binary_imgs.append(binary_imgs)
        return self.binary_imgs

    def blended_filter(self,img_list=None):
        img_list = self.images if img_list is None else img_list
        for img in img_list:
            sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
            sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
            self.blended.append(cv2.addWeighted(src1=sobelx, alpha=0.5, src2=sobely, beta=0.5, gamma=0))
        return self.blended

    def morphology_filter(self,img_list=None):
        img_list = self.blended if img_list is None else img_list
        for blended_img in img_list:
            self.morphology_imgs.append(cv2.morphologyEx(blended_img, cv2.MORPH_GRADIENT, np.ones((2,2), np.uint8)))
        return self.morphology_imgs

    def save_imgs(self,list_images,dir,pre_name='test_filters'):
        i = 1
        try:
            for img in list_images:
                cv2.imwrite(f'{dir}/{pre_name}_{i}.png', img)
                i+=1
            return True
        except:
            return False

