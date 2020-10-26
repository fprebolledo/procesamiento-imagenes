import cv2
import numpy as np

def find_point(rows, cols, binary, inverse):

    """ ESTA FUNCIÓN ENCUENTRA LA INTERSECCIÓN ENTRE LA FILA Y COLUMNA DONDE
    APARECEN LOS PRIMEROS PIXELES NEGROS PARA PODER RECORTAR LA IMAGEN DEL RELOJ,
    EL BOOLEANO INVERSE SIRVE PARA DECIR SI BUSCAMOS EL PUNTO DE ARRIBA A LA DERECHA
    O ABAJO A LA IZQUIERDA."""

    r = 0
    c = 0
    count = 0

    if not inverse:
        # Arriba a la derecha.
        r_r = range(rows)
        r_c = range(cols)
    else:
        # Abajo a la izquierda.
        r_r = range(rows-1, 0 , -1)
        r_c = range(cols-1, 0 , -1)

    #buscamos la fila donde se encuentran los primeros pixeles negros
    for row in r_r:
        for col in r_c:
            if binary[row][col] == 0:
                r+= row
                if count <3:
                    count += 1
                break
        if count == 3:
            break
    #buscamos la columna donde se encuentran los primeros pixeles negros
    count = 0
    for col in r_c:
        for row in r_r:
            if binary[row][col] == 0:
                c+= col
                if count <3:
                    count += 1
                break
        if count == 3:
            break
    
    # Intersección de ambos con un poco de corrección para no tomar pixeles
    # Blancos por accidente.

    if not inverse:
        r, c = int(r/3) + 10, int(c/3)
    else:
        r, c = int(r/3) - 50, int(c/3) - 25
    return r,c


def binarizar(gray, umbral, grande = True):
    if grande:
        lista = [2,3,5,6,9,10,12,15,18,21,24,25,27]
    else:
        lista = [2,3,4]

    # Se crea una imagen binaria con un umbral de 70
    _, binary = cv2.threshold(gray, umbral, 255, cv2.THRESH_BINARY)
    for i in lista:
        #iteramos en ventanas de ixi para sacar el ruido de la imagen
        # se hacen iteraciones para borrar primero los ruidos pequeños
        # y luego los más grandes que van quedando.
        kernel = np.ones((i,i),np.uint8)
        binary = cv2.morphologyEx(binary,cv2.MORPH_OPEN,kernel)
        binary = cv2.morphologyEx(binary,cv2.MORPH_CLOSE,kernel)

    return binary

def centro_masa(img, rows, cols):

    """FUNCIÓN QUE BUSCA EL CENTRO DE MASA EN LA SECCIÓN BLANCA DEL RELOJ RECORTADO"""

    r = 0
    c = 0
    area = 0

    for row in range(rows):
        for col in range(cols):
            if img[row][col] ==255:
                area += 1
                r += row
                c += col
    return int(r/area), int(c/area)


def find_radius(c_r, c_c,img):
    """ BUSCA EL LARGO DEL RADIO EN UN CIRCULO CON CENTRO C_R, C_C"""
    radius = 0
    
    while True:
        c_r += 1
        if img[c_r][c_c] == 0:
            break
        radius += 1
    return radius

def add_four(list_angles):

    """AÑADE 4 ANGULOS HACIA ARRIBA Y ABAJO PARA NO TENER QUE EL MINUTERO COINCIDA CON EL HORERO"""

    mini = min(list_angles)
    maxi = max(list_angles)

    for j in range(4):
        list_angles.append(mini - (j+1))
        list_angles.append(maxi + (j+1))

def find_angles(hour, minute, img, radius, c_r, c_c, color):
    """ ITERAMOS POR TODOS LOS ANGULOS POSIBLES CON UN RADIO DE 0.6*RADIO Y OTRO CON 0.3 * RADIO
    ESTO PARA DISINGUIR ENTRE EL MINUTERO Y EL HORERO, LUEGO SI COINCIDE CON PIXELES NEGROS ES 
    PORQUE ENCONTRÓ UNA MANIJA, A LA CUAL LE SACA EL PROMEDIO Y ENCUENTRA SU POSICIÓN DEACUERDO 
    A LAS COORDENADAS POLARES. """

    hour = hour*radius
    minute = minute*radius
    count = 0

    real_x = 0
    real_y = 0
    list_angles = []
    list_angles_hour = []
    list_angles_minute = []

    encontrados = 0
    for r in [minute, hour]:
        count = 0
        real_x = 0
        real_y = 0
        for angle in range(0, 360):
            sin = np.sin(angle * np.pi/180) 
            cos = np.cos(angle * np.pi/180) 
            x = int(c_c - round(r * cos))
            y = int(c_r - round(r * sin))
            if img[y][x] == 0 and angle not in list_angles:
                real_x += x
                real_y += y
                count += 1
                if r == minute:
                    list_angles_minute.append(angle)  
                else:
                    list_angles_hour.append(angle)                
                list_angles.append(angle)
            else:
                if count>0:
                    x = int(real_x/count)
                    y = int(real_y/count)
                    encontrados +=1
                    add_four(list_angles)
                    break

    if (real_x, real_y) == (0, 0):
        # Si no encontro la manija del horero (la más pequeña) en ninguna posición, quiere decir que 
        # está en la misma que el minutero. Por lo tanto le ponemos el mismo angulo.

        ang_hour, ang_min = int(np.mean(list_angles_minute)), int(np.mean(list_angles_minute))


    # CONVERSIÓN DEL ÁNGULO YA QUE DE 0 A 360 PARTE HORIZONTALMENTE.
    ang_hour, ang_min = int(np.mean(list_angles_hour)), int(np.mean(list_angles_minute))
    ang_hour, ang_min = 360 + (ang_hour-90), 360 + (ang_min-90)
    ang_hour, ang_min = ang_hour%360, ang_min%360
    return ang_hour, ang_min