import cv2
import numpy as np

patos = cv2.imread("IMG01.png")
lapiz = cv2.imread("IMG02.png")
flor = cv2.imread("IMG03a.png")
fondo = cv2.imread("IMG03b.png")

def print_img(img):
    cv2.imshow('Procesed image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def resize_img(img):
    # Achicamos la imagen a la mitad
    scale_percent = 50
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dsize = (width, height)
    output = cv2.resize(img, dsize)
    return output

def delete_blues(img_binary_rgb):
    x, y, _ = img_binary_rgb.shape
    img_new = img_binary_rgb.copy()
    for i in range(0,x):
        for j in range(0,y):
            if img_new[i][j][0] == 255 and img_new[i][j][1] == 0 and img_new[i][j][2] == 0:
                img_new[i][j][0], img_new[i][j][1], img_new[i][j][2] = 255, 255, 255
            if img_new[i][j][0] == 255 and img_new[i][j][1] == 255 and img_new[i][j][2] == 0:
                img_new[i][j][0], img_new[i][j][1], img_new[i][j][2] = 255, 255, 255
    return img_new

def encerrar_patos (img, img_raw, img_raw_raw):
    x, y = img.shape # aqui x son las filas y y son las columnas
    pato_2 = []
    pato_1 = []
    founded = 0
    col_con_algo = 0
    for col in reversed(range(0,y)):
        col_con_algo = 0
        cantidad_en_fila = 0 
        for row in range(0,x):
            if img[row][col] == 0:
                cantidad_en_fila += 1
                if cantidad_en_fila > 2:
                    col_con_algo = 1
                    if founded == 0:
                        founded = 1
                    if founded == 1:
                        pato_2.append((row,col))
                    if founded == 2:
                        pato_1.append((row,col))
        if col_con_algo == 0 and founded == 1:
            founded = 2
    row_min1 = np.min([tuplex[0] for tuplex in pato_1])
    row_max2 = np.max([tuplex[0] for tuplex in pato_1])
    col_min1 = np.min([tuplex[1] for tuplex in pato_1])
    col_max1 = np.max([tuplex[1] for tuplex in pato_1])
    cv2.rectangle(img_raw_raw, (col_min1*2-4, row_min1*2-4), (col_max1*2+8, row_max2*2-5), 160, 2)

    row_min1 = np.min([tuplex[0] for tuplex in pato_2])
    row_max2 = np.max([tuplex[0] for tuplex in pato_2])
    col_min1 = np.min([tuplex[1] for tuplex in pato_2])
    col_max1 = np.max([tuplex[1] for tuplex in pato_2])
    cv2.rectangle(img_raw_raw, (col_min1*2-4, row_min1*2-4), (col_max1*2+8, row_max2*2-5), 160, 2)
    # cv2.imwrite("IMG01_result.png", img_raw_raw)
    print_img(img_raw_raw)

def patos_function(img_raw):
    img_raw_raw = img_raw.copy()
    img_raw = resize_img(img_raw)
    img_binary = cv2.threshold(img_raw, 50, 255, cv2.THRESH_BINARY)[1]
    img_new = delete_blues(img_binary)
    gray = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)
    img_binary2 = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1]
    encerrar_patos(img_binary2, img_raw, img_raw_raw)
    
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
    patos_function(patos)