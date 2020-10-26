import numpy as np 
import cv2
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

######################### PROBLEMA 1 ########################
def func_1():
    #abrir datos
    f = open("covid_chile.txt", "r")
    datos = f.read()
    data = datos.split("\n")

    #restar media y selecciona 63 ultimos
    data = np.array(data[0:len(data)-1]).astype(int)
    f = data[len(data)-63: len(data)]

    promedio = round(np.mean(f),2)
    for i in range(len(f)):
        f[i]-= promedio

    #grafico de señal
    plt.plot(f)
    plt.ylabel('datos')
    plt.show()

    #transformada de fourier
    F = np.fft.fft(f)
    frecuencia = np.fft.fftfreq(63)
    # parte real
    xfabs=np.abs(F)
    # modulo = np.log(xfabs)

    plt.plot(frecuencia, xfabs)
    plt.ylabel('abs(F)')
    plt.show()

    """ En la gráfica podemos ver el máximo en 2 puntos, los cuales nos indican la mayor amplitud que tienen los datos pero en una escala diferente.
    Este máximo está situado en la frecuencia 0.144, por lo que podemos sacar el periodo => 6,94 , el cual redondearemos a 7.
    Esto nos dice que el periodo de la señal son 7 días aproximadamente. """

######################### Problema 2 #########################


def func_2():
    # Lectura de imagenes
    img_1 = cv2.imread('I1.png',0)
    img_2 = cv2.imread('I2.png',0)
    img_3 = cv2.imread('I3.png',0)

    #calcular fft
    f_1 = np.fft.fft2(img_1)
    f_2 = np.fft.fft2(img_2)
    f_3 = np.fft.fft2(img_3)


    #valor absoluto
    abs_1= np.abs(f_1)[:32,:32]
    abs_2 = np.abs(f_2)
    abs_3 = np.abs(f_3)[:32, :32]

    ## plot 

    fig = make_subplots(rows=1, cols=3,
                    specs=[[{'is_3d': True}, {'is_3d': True},  {'is_3d': True}]],
                    subplot_titles=['FFT I1', 'FFT I2', 'FFT I3'],
                    )

    fig.add_trace(go.Surface(z=abs_1), 1,1)

    fig.add_trace(go.Surface(z=abs_2), 1, 2)

    fig.add_trace(go.Surface(z=abs_3), 1, 3)

    fig.show()

    
    """
    Para la imagen 1:
    Podemos notar que en la transformada de fourier de la imagen 1 se ve un pico en la posicion (0,0) y otro pico más 
    pequeño en la posicion (16,0) pensamos que el pico en (0,0) es el que contiene la información de la imagen (fondo negro)
    y el pico en (16,0) es debido a que el ruido agregado a la imagen tiene un periodo de 16 con respecto al eje y.

    Para la imagen 2:
    Podemos notar que en la transformada de fourier de la imagen 2 se ve un pico en la posicion (0,0) y un pico más pequeño
    en la posicion (0,8) pensamos que el pico en (0,0) es el que contiene la información de la imagen (fondo negro)
    y el pico en (0,8) es debido a que el ruido agregado a la imagen tiene un periodo de 8 con respecto al eje x.

    Para la imagen 3: 
    Podemos notar que en la transformada de fourier de la imagen 3 se ve un pico en la posicion (0,0) y otro pico más
    pequeño en la posicion (0,8) y otro pico de igual tamaño en la posición (16,0), esta gráfica se debe a que se han mezclado las 2
    sinusoides anteriores, una con periodo 8 y la otra con periodo 16, sin embargo, estos picos son menores que los anteriores, esto
    probablemente se deba a que para formar esta imagen, se sacó el promedio de las 2 anteriores. También como mencionamos anteriormente
    el punto (0,0) tendría la información "real" de la imagen."""

################################ FUNC 3 #######################
def func_3():
    # Lectura de imagenes
    img_1 = cv2.imread('P.png',0)
    img_2 = cv2.imread('I2.png',0)

    #calcular fft
    f_1 = np.fft.fft2(img_1)
    f_2 = np.fft.fft2(img_2)


    #valor absoluto
    abs_1= np.abs(f_1)
    abs_2 = np.abs(f_2)

    # escala logaritmica
    log_1 = np.log(abs_1 + 1) [:32, :32]
    log_2 = np.log(abs_2 + 1) [:32, :32]
    
    log_1 = 255/np.max(log_1) * log_1
    log_2 = 255/np.max(log_2) * log_2

    ## plot 
    fig, a =  plt.subplots(2,2, figsize=(50,50))

    a[0][0].imshow(img_1, cmap = 'gray')
    a[0][0].set_title('entrada')
    a[0][1].imshow(log_1, cmap = "gray")
    a[0][1].set_title('FFT')

    a[1][0].imshow(img_2, cmap = 'gray')
    a[1][1].imshow(log_2, cmap = "gray")

    plt.show()
    """
    Podemos ver como en la transformada de fourier de la imagen con ruido periodico se ven dos puntos blancos, uno 
    en (0,0) y otro en (0,8). Tomando como ejemplo el ejercicio anterior esto significa que el periodo del ruido 
    es igual a 8. 
    Para quitar el ruido de la imagen se podrian buscar todas las coordenadas que tengan el color parecido a la coordenada (0,8) 
    (ya que según la imagen anterior no sabemos si es la única coordenada así) y dejarlas en negro, 
    de manera que al volver al dominio de la imagen se filtre la informacion que genera los ruidos.
    """
    
######################## Main #########################3
#func_1()
func_2()
#func_3()
