import cv2
import numpy as np
import helpers as p

##### VARIABLES GLOBALES #########################

FIRST_CROP = (700,600)
SECOND_CROP = (250,450)
MINUTE = 0.55
HOUR = 0.35
FILENAME = "img/IMG_04.jpg"

reals_hours = {"img/IMG_01.jpg": [6, 51], "img/IMG_02.jpg":[4, 10] , "img/IMG_03.jpg": [2,30] , 
"img/IMG_04.jpg": [12, 41], "img/IMG_05.jpg": [11, 14] , "img/IMG_06.jpg": [10, 8]}

############## LEER IMAGEN ########################
img = cv2.imread(FILENAME)
img = cv2.resize(img  , FIRST_CROP)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rows, cols = gray.shape

# Se crea una imagen binaria con un umbral de 70
binary = p.binarizar(gray, 70)


#calculamos los puntos bases del cuadrado del reloj
r1,c1 = p.find_point(rows, cols, binary, False)
r2,c2  = p.find_point(rows, cols, binary, True)

# recortamos la imagen a color, pasamos a gris 
# binarizamos denuevo ya que no tenemos el ruido de fondo de antes

gray = gray[r1:r2, c1:c2]
gray = cv2.resize(gray  , SECOND_CROP)
binary_final = p.binarizar(gray, 80, False)

#sacamos centro de masa
c_r, c_c = p.centro_masa(binary_final, 450, 250)

# ocupamos la binary sin manillas para encontrar el radio del circulo
binary  = binary[r1:r2, c1:c2]
binary = cv2.resize(binary  , SECOND_CROP)
radius = p.find_radius(c_r, c_c, binary)

# Imagen de color para ser mostrada
color  = img[r1:r2, c1:c2]
color = cv2.resize(color  , SECOND_CROP)


# Calculamos los angulos del horero y minutero y luego convertimos a la hora.
hour_angle, mini_angle = p.find_angles(HOUR, MINUTE, binary_final, radius, c_r, c_c, color)
hour = int(hour_angle/30) 
mini = int(mini_angle/6)
real = reals_hours[FILENAME]

if hour == 0:
    hour = 12
# Mostramos el resultado
print(f"H = {hour} M= {mini} Error_H = {abs(real[0]-hour)} Error_M={abs(real[1]-mini)}")