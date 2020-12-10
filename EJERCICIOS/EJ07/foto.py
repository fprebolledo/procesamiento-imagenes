import cv2

sunset = cv2.imread("img.jpg")
sunset = cv2.resize(sunset, (1920, 1280))
cv2.imwrite("imagen.jpg", sunset)
print(sunset.shape)
