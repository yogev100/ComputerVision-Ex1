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
from ex1_utils import LOAD_GRAY_SCALE
import cv2
import numpy as np


def main():
    gammaDisplay(('bac_con.png', LOAD_GRAY_SCALE))

def gammaDisplay(img_path:str, rep:int)->None:
    if(rep==2):
        img = cv2.imread(img_path)
    elif(rep==1):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    else:
        return
    temp=img
    cv2.imshow('Gamma Correction', img)

    def display(g: int): #inner function that gets the gamma number and display the new image
        g = float(g)
        g = g / 50
        img = np.uint8(255 * np.power((temp / 255), g))
        cv2.imshow('Gamma Correction',img)

    cv2.namedWindow('Gamma Correction')
    cv2.createTrackbar('Gamma','Gamma Correction',1,100,display)

    while(1):

        button=cv2.waitKey(1) & 0XFF
        if button == 27:
            break

    cv2.destroyAllWindows()




if __name__ == '__main__':
    main()
