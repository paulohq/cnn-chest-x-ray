import pandas as pd
import numpy as np
import os
from glob import glob
import random
import matplotlib.pyplot as plt
import cv2
import matplotlib.gridspec as gridspec
#import seaborn as sns
import zlib


PATH = '/home/paulo/mestrado/dataset/x-ray-chest/data-bbox-450'
SOURCE_IMAGES = PATH + '/data'
images = glob(os.path.join(SOURCE_IMAGES, "*.png"))

#print(images[0:10])
labels = pd.read_csv('/home/paulo/mestrado/dataset/x-ray-chest/BBox_List_2017.csv')
#print(labels.head(10))

#%matplotlib inline
image_name = images[0]
def plotImage(image_location):
    image = cv2.imread(image_name)
    image = cv2.resize(image, (512,512))
    plt.figure(0)
    #plt.plot(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    #plt.savefig('/tmp/imagem.png')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
    plt.axis('off')
    plt.savefig('/tmp/imagem.png')

#plotImage(image_name)


def plotMultipleImages():
    # Plot Multiple Images
    #xrays = glob('/kaggle/input/images_002/images/**')
    i_ = 0
    plt.rcParams['figure.figsize'] = (10.0, 10.0)
    plt.subplots_adjust(wspace=0, hspace=0)
    for l in images[:25]:
        im = cv2.imread(l)
        im = cv2.resize(im, (64, 64))
        plt.subplot(5, 5, i_+1) #.set_title(l)
        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        i_ += 1
    plt.show()
#plotMultipleImages()

def plotRandomImages():
    r = random.sample(images, 3)
    i = 1
    for l in r:
        image = cv2.imread(l)
        image = cv2.resize(image, (512, 512))
        #plt.figure(figsize=(16,16))
        plt.subplot(1,3, i)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        i += 1

    plt.show()

plotRandomImages()

def plotHistogram(a):
    """
    Plot histogram of RGB Pixel Intensities
    """
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title('Representative Image')
    b = cv2.resize(a, (512,512))
    plt.imshow(b)
    plt.axis('off')
    histo = plt.subplot(1,2,2)
    histo.set_ylabel('Count')
    histo.set_xlabel('Pixel Intensity')
    n_bins = 30
    plt.hist(a[:,:,0].flatten(), bins= n_bins, lw = 0, color='r', alpha=0.5);
    plt.hist(a[:,:,1].flatten(), bins= n_bins, lw = 0, color='g', alpha=0.5);
    plt.hist(a[:,:,2].flatten(), bins= n_bins, lw = 0, color='b', alpha=0.5);
#plotHistogram(X[1])