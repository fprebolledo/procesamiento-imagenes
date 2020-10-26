import cv2
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import math

estatua = cv2.imread("IMG01.png")
flor = cv2.imread("IMG02.png")
montana = cv2.imread("IMG03.png")

size = (400, 400)

def print_img(img, nombre):
    cv2.imshow(nombre, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def contraste(estatua):
    estatua = cv2.cvtColor(estatua, cv2.COLOR_BGR2GRAY)
    estatua = cv2.resize(estatua,size)
    grises = { i: [] for i in range(256)}
    for i in range(400):
        for j in range(400):
            grises[estatua[i][j]].append((i,j))

    posiciones = []
    for i in range(256):
        posiciones += grises[i]
    tamaño = int((400*400) / 256)
    
    color = 0
    for i in range(len(posiciones)):
        r,c = posiciones[i]
        estatua[r][c] = color
        if i%tamaño == 0:
            color += 1

    print_img(estatua, "contraste estutua")


def rotate_image(grados, img):
    matriz = np.zeros((400,400,3), np.uint8)
    print_img(matriz, "zeros")
    img = cv2.resize(img, size)

    sin = np.sin((grados * np.pi)/180) 
    cos = np.cos((grados * np.pi)/180) 
    for r in range(400):
        for c in range(400):
            a= r-200
            b = c-200
            i = round((a*cos) + (b* sin)) +200
            j = round(-(a*sin) + (b* cos)) +200
            if 0<=i<400 and 0<=j<400:
                matriz[r][c] = img[i][j]
    
    print_img(matriz, "rotacion imagen")
    return matriz

def make_mandala(img):
    grados = [40*i for i in range(9)]
    imagenes = []
    for grado in grados:
        im = rotate_image(grado, img)
        imagenes.append(im)
    
    final =  np.zeros((400,400,3), np.uint8)

    for r in range(400):
        for c in range(400):
            l1 = [ imagenes[i][r][c][0] for i in range(9)]
            l2 = [ imagenes[i][r][c][1] for i in range(9)] 
            l3 = [ imagenes[i][r][c][2] for i in range(9)] 
            ro = int(sum(l1) / 9)
            g = int(sum(l2) /9)
            b = int(sum(l3)/9)
            final[r][c] =  (ro,g, b)

    print_img(final, "mandala imagen")


def resize_img(img, valor):
    # Achicamos la imagen a la mitad
    scale_percent = valor
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dsize = (width, height)
    output = cv2.resize(img, dsize)
    return output

def bordes(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img= resize_img(img, 70)
    vertical_filter = np.array([[-1, 0, 1]], dtype=int)
    horizontal_filter = np.array([[-1, 0, 1]], dtype=int).T
    horizontal = cv2.filter2D(img, -1, horizontal_filter)
    vertical = cv2.filter2D(img, -1, vertical_filter)
    horizontal = cv2.threshold(horizontal, 60, 255, cv2.THRESH_BINARY)[1]
    vertical = cv2.threshold(vertical, 60, 255, cv2.THRESH_BINARY)[1]
    juntas = cv2.bitwise_or(horizontal, vertical)
    print_img(horizontal, "Bordes horizontales")
    print_img(vertical, "Bordes verticales")
    print_img(juntas, "Bordes horizontales y verticales")



if __name__ == "__main__":
    contraste(estatua)
    make_mandala(flor)
    bordes(montana)
