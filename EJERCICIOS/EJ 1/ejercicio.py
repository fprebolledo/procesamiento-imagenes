import cv2
import numpy as np

patos = cv2.imread("IMG01.png", cv2.IMREAD_GRAYSCALE)
lapiz = cv2.imread("IMG02.png")
flor = cv2.imread("IMG03a.png")
fondo = cv2.imread("IMG03b.png")

def print_img(img):
    cv2.imshow('Gray image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def find_point(rows, cols, binary, inverse, columna):
    r = 0
    c = 0
    count = 0
    # inverse es un booleano que dice si estamos buscando el punto
    # de arriba izquierda o abajo a la derrecha
    #invertimos según la variable
    if not inverse:
        r_r = range(rows)
        r_c = range(cols)
    else:
        r_r = range(rows-1, 0 , -1)
        r_c = range(cols-1, 0 , -1)

    if not columna:
        #buscamos la fila donde se encuentran los primeros pixeles negros
        for row in r_r:
            for col in r_c:
                if binary[row][col] == 255:
                    r+= row
                    c += col
                    if count <3:
                        count += 1
                    break
            if count == 3:
                break
        print("kjsddjjkad")
    else:
        #buscamos la columna donde se encuentran los primeros pixeles negros
        count = 0
        for col in r_c:
            for row in r_r:
                if binary[row][col] == 255:
                    c+= col
                    r += row
                    if count <3:
                        count += 1
                    break
            if count == 3:
                break
        
    if not inverse:
        r, c = int(r/3) , int(c/3)
    else:
        r, c = int(r/3) , int(c/3) 
    return r,c


def patos_find(patos_img):
    rows, cols,_ = patos_img.shape
    for row in range(rows):
        for col in range(cols):
            if patos_img[row][col]==[0,0,0]:
                patos_img[row][col] = 0
            else:
                patos_img[row][col] = 255
    
    print_img(patos_img)


def find_flor(flor_img, fondo_img):
    SECOND_CROP = (800,600)

    #hacemos un resize 
    flor_img = cv2.resize(flor_img, SECOND_CROP)
    fondo_img = cv2.resize(fondo_img, SECOND_CROP)
    gray = cv2.cvtColor(flor_img, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    lista = [2,3,4]
    for i in lista:
        #iteramos en ventanas de ixi para sacar el ruido de la imagen
        # se hacen iteraciones para borrar primero los ruidos pequeños
        # y luego los más grandes que van quedando.
        kernel = np.ones((i,i),np.uint8)
        binary = cv2.morphologyEx(binary,cv2.MORPH_OPEN,kernel)
        binary = cv2.morphologyEx(binary,cv2.MORPH_CLOSE,kernel)

    rows, cols,_ = flor_img.shape
    for row in range(rows-3):
        for col in range(cols-3):
            if binary[row][col] ==255:
                fondo_img[row][col] = flor_img[row-3][col-3]

    print_img(fondo_img)

def pencil(pencil_img):
    pencil_img = cv2.resize(pencil_img, (800,600))
    gray = cv2.cvtColor(pencil_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)

    rows, cols = binary.shape
    r1, c1 = find_point(rows, cols, binary, False, True)
    r2, c2 = find_point(rows, cols, binary, True, False)
    r3, c3 = r2, c1

    ########## BORRAR CUANDO ESTE LISTO ########33
    cv2.line(pencil_img,(c1,r1),(c3,r3),(255,255,0),4)
    cv2.line(pencil_img,(c3,r3),(c2,r2),(255,255,0),4)
    cv2.line(pencil_img,(c2,r2),(c1,r1),(255,255,0),4)
    print_img(pencil_img)
    ################################################

    cateto1 = r3-r1 
    cateto2 = c2-c3
    r2 = cateto1**2 + cateto2**2
    dist = round(r2**0.5)


    ##### PARA PONER EL TEXTO EN LA IMAGEN #####
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (80,400)
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2

    cv2.putText(binary,f'd = {dist} ', 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)
    print_img(binary)

if __name__ == "__main__":
    pencil(lapiz)
    find_flor(flor, fondo)