import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal


PARAMETROS= {"img1": [[(55.5,32), (52,768), (108, 631), (164,8), (162, 32),(35.5,738.5)], 44, 12], 
            "img2": [[(770.5,35.5), (26, 22), (687,58), (628, 129), 
                    (120, 165), (68, 121)], 54,8],
            "img3": [[(6.5,31.5), (786.5, 2), (736, 38),(667,44)], 50,8],
            "img4": [[(19.5,52.5), (156, 28), (748.5,19.5),(772,156)], 60,9],

            "img5": [[(13.5,17.5),(26.5,35.5),(39.5, 52.5), (52.5,69.5), (65.5,82.5), 
                    (40,17), (52, 35), (65, 52), (78, 69),(91, 82),
                    (66,17), (79, 35), (92, 52), (105, 69),(118, 82),
                    (786,17),(773,35),(760, 52), (747,69), (734,82), 
                    (760,17), (747, 35), (734, 52), (721, 69),(708, 82),
                    (734,17), (721, 35), (708, 52), (695, 69),(682, 82),
                    (13, 195), (26, 178), (39, 161), (52, 144),(65, 127),
                    (40,195), (52,178), (65,161), (78,144), (91,127) ,
                    (66,195), (79,178), (92,161), (105,144), (118,127),
                    (600,17), (613, 35), (626, 52), (639, 69),(652, 82),
                    (600,195), (613,178), (626,161), (639,144), (652,127),
                    (627,17), (640, 35), (653, 52), (666, 69),(679, 82),
                    (627,195), (640,178), (653,161), (666,144), (679,127),
                    (654,195), (667,178), (680,161), (693,144),
                    (654,17), (667, 35), (680, 52), (693, 69),
                    (199,195), (186,178), (173,161), (160,144), (147,127),
                    (199,17), (186,35), (173,52), (160,69), (147,82),
                    (172,17), (172,195), (159,35), (159,178), (146,52), 
                    (146,161), (133,69), (133,144), (120,82), (120,127), 
                    (146,17), (146,195), (133,35), (133,178), (120,52), (120,161),
                    (786,195), (773,178), (760,161), (747,144), (734,127),
                    (734,195), (721,178), (708,161), (695,144), (682,127), 
                    (760,195), (747,178), (734,161), (721,144), (708,127)
                    ], 12, 3]
                    }

def gkern(kernlen=21, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d


def print_fourier(fft):
    f_1 = np.log(np.abs(fft)+1)
    plt.imshow( f_1, cmap="gray")
    plt.show()

def print_img(img):
    plt.imshow(img, cmap="gray")
    plt.show()

def not_in_range(f, dif, limit):
    lista = [f-dif, f+ dif]
    if lista[0] <0 :
        return (True, 1)
    elif lista[1]>limit:
        return (True, 0)
    else:
        return (False, None)

def filter_img(img_path, frecuencies_list, len_kernel, std, res_path):
    img = cv2.imread(img_path,0)
    size = img.shape

    ### Calcular fourier y kernel de filtro gaus y print
    fft = np.fft.fft2(img)
    kernel = gkern(len_kernel, std)
    fondo = np.ones(size)
    dif = int(len_kernel/2)
    var = 1
    kernel2 = 1-signal.gaussian(var*2, std=1).reshape(var*2, 1)
    
    
    for f_row, f_col in frecuencies_list:
        dif_r1, dif_r2, dif_c1, dif_c2 = (dif, dif, dif, dif)
        r_r, r_c = int(size[0]-int(f_row)), int(size[1]-int(f_col))

        ## revisar si se puede aplicar filtro de inmediato o si queda fuera de rango
        rango_row = not_in_range(int(f_row), dif, size[0])
        rango_col = not_in_range(int(f_col), dif, size[1])

        # queda fuera, entonces rehacer diferencias.
        if rango_row[0]:
            if rango_row[1] == 1:
                dif_r1 = int(f_row) - 0
            else:
                dif_r2 = size[0]- int(f_row)

        if rango_col[0]:
            if rango_col[1] == 1:
                dif_c1 = int(f_col) - 0
            else:
                dif_c2 = size[1]- int(f_col)

        #aplicar los filtros de gauss
        
        fondo[int(f_row)-dif_r1: int(f_row)+dif_r2, int(f_col)-dif_c1: int(f_col)+dif_c2] = 1-kernel[dif-dif_r1: dif+dif_r2, dif-dif_c1:dif+dif_c2]
        fondo[r_r-dif_r2: r_r+dif_r1, r_c-dif_c2: r_c+dif_c1] = 1-kernel[dif-dif_r2: dif+dif_r1, dif-dif_c2:dif+dif_c1]

   
    filtrado = fondo*fft

    cv2.imwrite("hola.png",np.abs(np.fft.fftshift(fondo*np.log(np.abs(fft)+1))))  
    inversa = np.fft.ifft2(filtrado)

    cv2.imwrite(res_path,np.abs(inversa))  

for i in range(1,6):
    filter_img(f"Imagenes/J{i}.png", *PARAMETROS[f"img{i}"], f"resultados/J{i}.png")