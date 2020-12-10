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
    print_img(tazmania, "original")
    kernel = np.ones((5, 5), np.uint8) 
    gradiente = cv2.morphologyEx(tazmania, cv2.MORPH_GRADIENT, kernel)
    gradiente = np.abs(gradiente)
    cv2.imwrite("gradiente.png", gradiente)
    print_img(gradiente, "gradiente")

def drops_segmentations(drops):
    print_img(drops,"original")
    kernel=  np.ones((60, 60), np.uint8) 
    tophat = cv2.morphologyEx(drops, cv2.MORPH_TOPHAT, kernel)
    tophat = cv2.cvtColor(tophat, cv2.COLOR_BGR2GRAY)
    _, threshed = cv2.threshold(tophat,70,255,cv2.THRESH_BINARY)
    kernel2 = np.ones((9, 9), np.uint8)
    eroded = cv2.erode(threshed, kernel2)
    dilated = cv2.dilate(eroded, kernel2)
    kernel3 = np.ones((3, 3), np.uint8)
    eroded2 = cv2.erode(dilated, kernel3)
    border = dilated-eroded2
    border = border == 255 ## creo una mascara booleana con true donde es 255
    drops[border] = (0,0,255)
    print_img(drops,"segmentada")
    

if __name__ == "__main__":
    sunset_segmentation(sunset)
    morph_gradient(tazmania)
    drops_segmentations(drops)
