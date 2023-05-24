# This is a sample Python script.
import matplotlib.pyplot as plt

from filters import ImageFilter
import cv2
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    img_filter = ImageFilter()
    img_filter.read_imgs()
    img_filter.hls_filter()
    img_filter.binary_filter()
    img_filter.blended_filter()
    img_filter.morphology_filter()
    print(img_filter.binary_imgs[0])
