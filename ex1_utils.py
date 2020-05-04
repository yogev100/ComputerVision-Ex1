"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
import math
from typing import List
import matplotlib.pyplot as plt

import numpy as np
import cv2
LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:

    return 205836927


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    if (representation == 1):
        plt.gray() #change the presentation mode to grayscale
        img_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) # initial the image in grayscale
        img_gray = cv2.normalize(img_gray.astype('double'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        return img_gray
    if (representation == 2):
        img = cv2.imread(filename)
        img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # initial the image in rgb
        img_color = cv2.normalize(img_color.astype('double'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        return img_color


def imDisplay(filename: str, representation: int):
    if (representation == 1):
        plt.imshow(imReadAndConvert(filename, 1)) # show the image in grayscale
        plt.show()
    if (representation == 2):
        plt.imshow(imReadAndConvert(filename, 2)) # show the image in rgb
        plt.show()


def transformRGB2YIQ(imRGB: np.ndarray) -> np.ndarray:
    if(imRGB.ndim==3):# if the image is in rgb colors
        row = len(imRGB)
        col = len(imRGB[0])
        z=len(imRGB[0][0])
        array =np.array([[0.299, 0.587, 0.114],[0.596, -0.275, -0.321],[0.212, -0.523, 0.311]])
        rgb_trans=imRGB.transpose() # transpose the matrix for multiply it
        yiq_mul=array.dot(rgb_trans.reshape(3,row*col))# the matrix after multiply
        yiq_before=np.reshape(yiq_mul,(z,col,row))# yiq before reshape
        yiq=np.transpose(yiq_before) # the original yiq
        return yiq
    else: # if the image is in grayscale
        return imRGB

#######----------4.3-----------########
def transformYIQ2RGB(imYIQ: np.ndarray) -> np.ndarray:
    if(imYIQ.ndim==3): # if the image is in rgb colors
        row = len(imYIQ)
        col = len(imYIQ[0])
        z = len(imYIQ[0][0])
        array =np.array([[0.299, 0.587, 0.114],[0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
        array=np.linalg.inv(array)# the opposite matrix
        yiq_trans = imYIQ.transpose()
        rgb_mul = array.dot(yiq_trans.reshape(3, row*col)) # multiply the array with the new form of yiq_trans
        rgb_before = np.reshape(rgb_mul, (z, col, row)) #rgb before reshape
        rgb=np.transpose(rgb_before)# the original rgb
        return rgb
    else: # if the image is in grayscale
        return imYIQ


def hsitogramEqualize(imOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    if (imOrig.ndim == 2): # if the image is in grayscale
        imgOrig = cv2.normalize(imOrig.astype('float64'), None, 0, 255, cv2.NORM_MINMAX)
        y_channel = imgOrig

    else: # if the image is in rgb colors
        imgOrig = cv2.normalize(imOrig.astype('float64'), None, 0, 255, cv2.NORM_MINMAX)
        y_channel = transformRGB2YIQ(imgOrig)[:, :, 0] # take the y channel

    histogram = calHist(y_channel)  # step 1 - calculate the histogram image
    cum_sum = calCumSum(histogram)  # step 2 - calculate the cumulative sum
    nor_cum_sum = cum_sum / cum_sum.max()  # step 3 - normalazied the cum sum
    map_img = nor_cum_sum * 255  # step 4 - mapping the old intensity colors to new intensity color
    round_map = map_img.astype('uint8')  # step 5 - round the values
    old_y_channel = np.array(y_channel).astype('uint8') # casting to int for set each value at the round_map
    new_img = round_map[old_y_channel]  # step 6 - set the new intensity value according to the map
    imgOrig = cv2.normalize(imgOrig.astype('double'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    histogram_new = calHist(new_img) # the new image's histogram

    if(imOrig.ndim==2):
        return new_img, histogram, histogram_new

    else :
        yiq = transformRGB2YIQ(imgOrig)
        yiq[:, :, 0] = new_img / 255
        rgb = transformYIQ2RGB(yiq)
        return rgb, histogram, histogram_new

def calHist(img:np.ndarray)->np.ndarray: # side function that calculate the number each intensity color and return an array
    img_flat=img.ravel() # flat the array for running in one for
    hist=np.zeros(256)
    for pix in img_flat:
        pix=math.floor(pix)
        hist[pix]+=1

    return hist

def calCumSum(arr:np.ndarray)->np.ndarray:
    cum_sum=np.zeros_like(arr)
    cum_sum[0]=arr[0]
    for i in range(1,len(arr)):
        cum_sum[i]=cum_sum[i-1]+arr[i]

    return cum_sum

def quantizeImage(imOrig:np.ndarray, nQuant:int, nIter:int)->(List[np.ndarray],List[float]):
    images=[] # the list of images
    errors=[] # the list of error
    num_dim=imOrig.ndim # rgb/grayscale
    if (num_dim == 3):# extract the y channel
        y_channel= transformRGB2YIQ(imOrig)[:, :, 0]
    else :
        y_channel=np.copy(imOrig)

    histogram=calHist(y_channel*255)
    k=256/nQuant #initial ranges
    k=math.floor(k) # round the k number
    Q=np.zeros(nQuant) # initial array of weighted average in each range
    Z=np.arange(nQuant+1) # initial array of boundries
    Z=Z*k # create uniform range
    Z[len(Z)-1]=255 # the last bound is 255
    for n in range(0,nIter):
        Q=q_update(Q,Z,histogram,nQuant)# update the values in each range
        new_y_channel = update_y_channel(Z, Q, imOrig) # update the y channel according to the new values in Z and Q arrays

        if(num_dim==3): # insert the new quantization image
            new_img=get_new_image(new_y_channel, imOrig)
            images.insert(n, new_img)
        else:
            images.insert(n,(new_y_channel*255))

        error=np.mean(np.abs(new_y_channel*255- y_channel*255)) #calculate the value of the error
        errors.insert(n, error)

        zArr=z_update(Z,Q,nQuant) # update the z array

    return images,errors

def update_y_channel(Z:np.ndarray,Q:np.ndarray,img:np.ndarray)->np.ndarray:
    if(img.ndim==3):
        y_channel = np.copy(transformRGB2YIQ(img)[:, :, 0])*255
    else :
        y_channel = np.copy(img)*255
    for z in range(0, len(Z) - 1):
        y_channel[(y_channel < Z[z + 1]) & (y_channel >= Z[z])] = math.floor(Q[z])

    return y_channel/255

def z_update(zArr:np.ndarray,qArr:np.ndarray,nQuant:int)->np.ndarray:
    for t in range(1, nQuant - 1):
        zArr[t] = math.floor((qArr[t - 1] + qArr[t])) / 2  # compute the new z_i
    return zArr

def q_update(q_arr:np.ndarray,z_arr:np.ndarray,hist:np.ndarray,nQuant:int)->np.ndarray:
    for i in range(0, nQuant):
        q_i = getWeightedMean(hist, z_arr[i], z_arr[i + 1])  # compute the value (q_i)
        q_arr[i] = q_i

    return q_arr

def get_new_image(y_channel:np.ndarray,imOrig:np.ndarray)->np.ndarray:
    yiq=transformRGB2YIQ(imOrig) #transform to yiq
    if(imOrig.ndim==3): # rgb image
        yiq[:,:,0]=y_channel
    else: # grayscale image
        imOrig=y_channel
        return imOrig

    rgb=transformYIQ2RGB(yiq) # return to rgb
    return rgb

def getWeightedMean(hist:np.ndarray,x:int,y:int)->int:
    sum1=0
    sum2=0
    for b in range(x,y):
        sum1+=hist[b]*b
        sum2+=hist[b]
    num=sum1/sum2
    return num
