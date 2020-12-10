import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import dct, fft
from skimage import color, data, restoration

def gkern(l=5, sig=1.):
    """
    creates gaussian kernel with side length l and a sigma of sig
    """
    # código sacado de : https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))

    return kernel / np.sum(kernel)

def calculate_largue(maximo, grupos):
    return maximo-grupos + 1
    
def degradacion_vertical(path, n):
    img = cv2.imread(f'{path}', cv2.IMREAD_GRAYSCALE)
    shape = img.shape
    M = shape[1]
    N = calculate_largue(shape[0], n)
    H = np.zeros((N, M))
    h_row = [1/n]*n
    for i in range(N):
        H[i][i:n+i] = h_row

    result = np.round(H.dot(img))
    return img, result, H

def find_A(H,W):
    lammbda = 10**6
    first =lammbda * (np.transpose(H).dot(H))
    second = np.transpose(W).dot(W)
    A = lammbda*np.linalg.inv(first + second)
    A = A.dot(np.transpose(H))
    return A

def restauracion_normal(degradacion, H, tipo = "vertical"):
    shape = H.shape #antes degradacion.shape
    W = np.identity(shape[1])
    A = find_A(H,W)
    if tipo == "vertical":
        restauracion = A.dot(degradacion)
    elif tipo == "c":
        restauracion = np.transpose(A).dot(degradacion)
    else:    
        restauracion = A.dot(degradacion.flatten())
        restauracion = np.reshape(restauracion, (64,64))
    return restauracion

def calcular_error(original, restauracion):
    error = np.round(np.average((np.abs(original-restauracion)/255)*100), 2)
    print("el error es:", error)
    plt.imshow(restauracion, cmap="gray")
    plt.show()

def restauracion_minio(degradacion, H):
    shape = H.shape
    N = shape[0]
    P = np.identity(shape[0])
    for _ in range(shape[1]-N):
        P = np.insert(P, P.shape[1], np.array([0]*shape[0]), 1)
    
    W = P-H
    A = find_A(H,W)
    restauracion = A.dot(degradacion)
    return restauracion

def restauracion_dct(degradacion, H, n):
    M = H.shape[1]
    K = np.identity(M)
    D = dct(np.eye(M), axis=0)
    kernel = gkern(2*n, n)
    
    K[0:n, 0:n] *= kernel[0:n, 0:n]
    W = K.dot(D)
    A = find_A(H,W)
    restauracion = A.dot(degradacion)
    return restauracion

def degradacion_foco(path, n):
    img = cv2.imread(f'{path}', cv2.IMREAD_GRAYSCALE)
    img_columnizada = []
    M, C = img.shape[0], img.shape[1]
    N = calculate_largue(M, n)
    img_columnizada = img.flatten()
    H = []
    fila = [1/(n*n)]*n + [0]*(M-n) 
    zeros = [0]*M
    fila = fila*n+ zeros*(C-n)
    for _ in range(0, N**2, N):
        for _ in range(N):
            copia = fila.copy()
            H.append(copia)
            element = fila.pop(M**2-1)
            fila.insert(0, element)
        
        for _ in range(M-N):
            element = fila.pop(M**2-1)
            fila.insert(0, element)

    H = np.array(H)
    G = H.dot(img_columnizada)
    G = G.reshape((N, N))
    return img, G, H

def calculate_H_from_img(img, n):
    N, _ = img.shape
    M = N+n-1
    H = np.zeros((N, M))
    h_row = [1/n]*n

    for i in range(N):
        H[i][i:n+i] = h_row
    return H

def restore_img_dct(img, n, k):
    B, G, R = img[:,:,2], img[:,:,1], img[:,:,0]
    B, G, R = np.transpose(B), np.transpose(G), np.transpose(R)
    H = calculate_H_from_img(R, n)
    rest_R = restauracion_dct(R, H,k)
    rest_G = restauracion_dct(G, H,k)
    rest_B = restauracion_dct(B, H,k)
    rest_R, rest_G, rest_B = rest_R.astype(np.float32)/255, rest_G.astype(np.float32)/255, rest_B.astype(np.float32)/255
    restored = cv2.merge([np.transpose(rest_B), np.transpose(rest_G), np.transpose(rest_R)])
    return restored

def restore_img_minio(img, n):
    B, G, R = img[:,:,2], img[:,:,1], img[:,:,0]
    B, G, R = np.transpose(B), np.transpose(G), np.transpose(R)
    H = calculate_H_from_img(R, n)
    rest_R = restauracion_minio(R, H)
    rest_G = restauracion_minio(G, H)
    rest_B = restauracion_minio(B, H)
    rest_R, rest_G, rest_B = rest_R.astype(np.float32)/255, rest_G.astype(np.float32)/255, rest_B.astype(np.float32)/255
    restored = cv2.merge([np.transpose(rest_B), np.transpose(rest_G), np.transpose(rest_R)])
    return restored


if __name__ == "__main__":
    original, degradacion, H = degradacion_vertical("santiago512.png", 57)
    print("------------------NORMAL--------------------------")
    restauracion1 = restauracion_normal(degradacion, H)
    calcular_error(original, restauracion1)
    print("-----------------MINIO-------------------------")

    restauracion2 = restauracion_minio(degradacion, H)
    calcular_error(original, restauracion2)
    print("---------------- DCT--------------------")
    rest3 = restauracion_dct(degradacion, H, 8)
    calcular_error(original, rest3)
    
    print("---------------- moon ----------------------")
    original, degradacion, H = degradacion_foco("moon64.png", 5)
    plt.imshow(degradacion)
    plt.show()
    rest1 = restauracion_normal(degradacion, H, "blur")
    calcular_error(original, rest1)

    print("---------------- Proceso real 1 ----------------------")
    n = 50
    img2 = cv2.imread('image_blur_01.png')
    rest = restore_img_dct(img2, n, 12)
    plt.imshow(rest)
    plt.show()
    
    print("---------------- Proceso real 2 ----------------------")
    n2 = 10
    rest = restore_img_minio(img2, n2)
    plt.imshow(rest)
    plt.show()
