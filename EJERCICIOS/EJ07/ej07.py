import cv2
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import math
import itertools

sunset = cv2.imread("atardecer.png")
drops = cv2.imread("gotas.png")
tazmania = cv2.imread("tazmania.png")

def print_img(img, nombre):
    cv2.imshow(nombre, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def sunset_segmentation(sunset):
    print(sunset.shape)
    # saco la gaviota
    sun = cv2.medianBlur(sunset,5)
    print_img(sun, "media")

    #saco la luna.
    kernel = np.ones((5, 5), np.uint8) 
    gav = cv2.erode(sunset, kernel)  
    for i in range(8):
        gav = cv2.erode(gav, kernel)  

    gav = cv2.dilate(gav,kernel,iterations = 8) 
    # para sacar el perimetro de cada una
    kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8) 
    sun_erode = cv2.erode(sun, kernel)  
    gav_erode = cv2.erode(gav, kernel)
    segm_gav = gav-gav_erode
    segm_sun = sun-sun_erode

    _, thresh1 = cv2.threshold(segm_gav,15,255,cv2.THRESH_BINARY)
    _, thresh2 = cv2.threshold(segm_sun,30,255,cv2.THRESH_BINARY)
    tresholding_img = thresh2+thresh1
    pixels = np.where(tresholding_img == 255)
    permutaciones = list(set(itertools.permutations([-1, -1, 1, 1, 0], 2)))
    for r,c in zip(pixels[0], pixels[1]):
        for mr, mc in permutaciones:

            sunset[r+mr][c+mc] = (255,255,0)
    print_img(sunset, "gaviota perimtro")

def morph_gradient(tazmania):
    kernel = np.ones((5, 5), np.uint8) 
    gradiente = cv2.morphologyEx(tazmania, cv2.MORPH_GRADIENT, kernel)
    gradiente = np.abs(gradiente)
    cv2.imwrite("gradiente.png", gradiente)
    print_img(gradiente, "tazmania")

def drops_segmentations(drops):
    kernel=  np.ones((30, 30), np.uint8) 
    tophat = cv2.morphologyEx(drops, cv2.MORPH_TOPHAT, kernel)
    print_img(tophat, "tophat")
    
if __name__ == "__main__":
    # sunset_segmentation(sunset)
    # morph_gradient(tazmania)
    drops_segmentations(drops)