# This is a sample Python script.
import matplotlib.pyplot as plt
from filters import ImageFilter
import cv2

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    img_filter = ImageFilter()
    images = img_filter.read_imgs()
    hls = img_filter.hls_filter()[0]
    blended = img_filter.blended_filter()[0]
    morphology = img_filter.morphology_filter()[0]
    binary = img_filter.binary_filter(images)[0]
    gray = img_filter.gray_filter()[0]
    sobel = img_filter.sobel_filter()[0]

    img_filter.save_imgs([hls,blended,morphology,binary,gray,sobel], 'resultImages')
    plt.imshow(sobel)
    plt.show()
