import cv2
import numpy as np
import math

def pHash(img):
    #加载并调整图片为32x32灰度图片
    img = img[0].swapaxes(0,2).swapaxes(0,1)
    img=cv2.resize(img,(32,32),interpolation=cv2.INTER_CUBIC)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
 
   #离散余弦变换
    img = cv2.dct(img)
    img = img[0:8,0:8]
    avg = 0
    hash_str = ''
 
    #计算均值
    for i in range(8):
        for j in range(8):
            avg += img[i,j]
    avg = avg/64
 
    #获得hsah
    for i in range(8):
        for j in range(8):
            if  img[i,j]>avg:
                hash_str=hash_str+'1'
            else:
                hash_str=hash_str+'0'            
    return hash_str
 
def cmpHash(hash1,hash2):
    n=0
    if len(hash1)!=len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i]!=hash2[i]:
            n=n+1
    return n

def count_phash(img1,img2):
    # return hmdistance(phash(np.array(img1)),phash(np.array(img2)))
    return cmpHash(pHash(np.array(img1)), pHash(np.array(img2)))

def get_median(data):
    data.sort()
    half = len(data) // 2
    return (data[half] + data[~half]) / 2


def psnr(img1, img2):
    img1 = np.float64(img1)
    img2 = np.float64(img2)
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
