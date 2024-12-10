import cv2
import numpy as np

original_image = cv2.imread('b.jpeg', cv2.IMREAD_GRAYSCALE)
original_matrix = original_image.copy()

dft = cv2.dft(np.float32(original_matrix), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# print(dft_shift.shape)