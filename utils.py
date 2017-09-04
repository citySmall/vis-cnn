from matplotlib import pylab as plt
import numpy as np
from PIL import Image
import cv2
import skimage
import skimage.io
import skimage.transform

def smooth_50center_crop(img_path, img_width=224, img_height=224, t='TRIAN', is_color=True):
    """
    crop image from center
    """
    if is_color:
        img = skimage.io.imread(img_path)
        # img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = img[:img.shape[0] - 50, :]

        # using threshold to remove the noise and using GaussianBlur to smooth grayscale image
        ret, img = cv2.threshold(img, 80, 255, cv2.THRESH_TOZERO)
        img = cv2.GaussianBlur(img, (5, 5), 0)

        plt.imshow(img)
        plt.show()

        short_edge = min(img.shape[:2])
        yy = int((img.shape[0] - short_edge) / 2)
        xx = int((img.shape[1] - short_edge) / 2)
        img = img[yy: yy + short_edge, xx: xx + short_edge]
        plt.imshow(img)
        plt.show()
        img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)
        plt.imshow(img)
        plt.show()
    else:
        # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = skimage.io.imread(img_path, as_grey=True)
        img = img[50:, :]

        # using threshold to remove the noise and using GaussianBlur to smooth grayscale image
        ret, img = cv2.threshold(img, 100, 255, cv2.THRESH_TOZERO)
        img = cv2.GaussianBlur(img, (3, 3), 0)

        short_edge = min(img.shape[:2])
        yy = int((img.shape[0] - short_edge) / 2)
        xx = int((img.shape[1] - short_edge) / 2)
        img = img[yy: yy + short_edge, xx: xx + short_edge]
        img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)
        img = img.reshape((img.shape[0], img.shape[1], 1))
    return img


def show_image(image, grayscale = True, ax=None, title=''):
    if ax is None:
        plt.figure()
    plt.axis('off')
    
    if len(image.shape) == 2 or grayscale == True:
        if len(image.shape) == 3:
            image = np.sum(np.abs(image), axis=2)
        vmax = np.percentile(image, 99)
        vmin = np.min(image)

        plt.imshow(image, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
        plt.title(title)
    else:
        #image = image + 127.5
        #image = image.astype('uint8')
        
        plt.imshow(image)
        plt.title(title)
    #plt.show()
    
def load_image(file_path):
    #im = Image.open(file_path).resize((299,299), Image.ANTIALIAS)
    #im = Image.open(file_path)
    #im = np.asarray(im)
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    print(img.shape)
    img = img.reshape((img.shape[0], img.shape[1], 1))
    print(img.shape)
    return img - 127.5