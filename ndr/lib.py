import tensorflow as tf
import numpy as np
from astropy.io import fits
from os import listdir
from os.path import isfile, join
from scipy import ndimage


def GetFiles(folder, include_full_path=True):
    files = []
    for file in listdir(folder):
        full_path = join(folder, file)
        if isfile(full_path):
            files.append(full_path if include_full_path else file)
    return files


def GetLabel(file):
    index = file.find("clock_")
    label = np.zeros(10)
    label[int(file[index + 6])] = 1.0
    label.astype('float32')
    return label


def GetData(file):
    hdulist = fits.open(file)
    data = hdulist[0].data
    hdulist.close()
    data.shape = (30, 30)
    indices = 29 - np.arange(30)
    data = data[indices, :]
    data.shape = (900)
    return data

def GetImage(data, index):
    img = data[index, :]
    img.shape = (30, 30)
    return img

def FisherScore(l1, l2, lc, r1, r2, rc):
    d = np.power(l1 - r1, 2.0)
    lv = 0
    if lc > 0:
        lm = l1 / lc
        lv = l2 - lm * lm * lc
    rv = 0
    if rc > 0:
        rm = r1 / rc
        rv = r2 - rm * rm * rc
    ret = d / (lv + rv)
    return ret


def RemoveBkgd(img, s=3.0):
    img = img * 1.0
    img2 = img - ndimage.gaussian_filter(img, sigma=s, mode='nearest')
    data = np.copy(img2)
    data.shape = (data.size)
    data = np.sort(data)
    l1 = 0.0
    l2 = 0.0
    lc = 0
    r1 = np.sum(data)
    r2 = np.sum(np.power(data, 2.0))
    rc = data.size
    max_score = 0
    max_v = 0
    for i in range(0, data.size):
        v = data[i]
        v2 = np.power(v, 2.0)
        l1 += v
        l2 += v2
        lc += 1
        r1 -= v
        r2 -= v2
        rc -= 1
        score = FisherScore(l1, l2, lc, r1, r2, rc)
        if (score > max_score):
            max_score = score
            max_v = v
    img2 = np.greater(img2, max_v) * img2
    return img2

def CentralizeImage(img):
    img = img * 1.0
    num_rows = img.shape[0]
    num_cols = img.shape[1]
    tmp_x = np.array(range(0, img.size))
    tmp_x.shape = img.shape
    tmp_y = np.copy(tmp_x)
    tmp_x = tmp_x % num_cols
    tmp_y = tmp_y / num_cols
    img_sum = np.sum(img)
    mean_x = int(np.sum(tmp_x * img) / img_sum) - num_cols / 2
    mean_y = int(np.sum(tmp_y * img) / img_sum) - num_rows / 2
    img = np.roll(img, -mean_y, axis=0)
    img = np.roll(img, -mean_x, axis=1)
    return img

def NormalizeImage(img):
    img = img * 1.0
    img.shape = (28, 28)
    img = RemoveBkgd(img)
    img = CentralizeImage(img)
    img.shape = (784)
    img /= np.max(img)
    img.astype('float32')
    return img

def NormalizeImages(imgs):
    new_imgs = []
    for img in imgs:
        img = NormalizeImage(img)
        img /= np.max(img)
        new_imgs.append(img)
    return np.array(new_imgs)

def LoadData(folder):
    features = []
    labels = []
    files = GetFiles(folder)
    print "Total files: ", len(files)
    for file in files:
        label = GetLabel(file)
        print len(labels), label, file
        labels.append(label)
        img = GetData(file) * 1.0
        img.shape = (30, 30)
        img = np.array(img[1:29, 1:29])
        img.shape = (784)
        img.astype('float32')
        features.append(img)
    features = np.array(features)
    labels = np.array(labels)
    return labels, features
