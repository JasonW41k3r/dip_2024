import cv2
import numpy as np

T, a, b = 1, 0.1, 0.1

# 运动模糊滤波器
def filter_1(shape, T, a, b):
    height, width = shape
    H = np.zeros((height, width, 2), dtype=np.float32)
    step_u = height // 2
    step_v = width // 2
    for u in range(height):
        for v in range(width):
            u_shift = u - step_u
            v_shift = v - step_v
            temp = u_shift * a + v_shift * b
            if temp == 0:
                value = T
            else:
                value = T / (np.pi * temp) * np.sin(np.pi * temp) * np.exp(-1j * np.pi * temp)
            H[u][v][0] = value.real
            H[u][v][1] = value.imag
    return H

# 维纳滤波器
def filter_2(K, H):
    height, width = H.shape[0], H.shape[1]
    F = np.zeros((height, width, 2), dtype=np.float32)
    for u in range(height):
        for v in range(width):
            pow_sum = (H[u, v, 0] ** 2 + H[u, v, 1] ** 2)
            temp = pow_sum / (pow_sum + K)
            value_real = temp * H[u, v, 0]
            value_imag = -temp * H[u, v, 1]
            F[u, v, 0] = value_real
            F[u, v, 1] = value_imag
    return F

image = cv2.imread('b.jpeg', cv2.IMREAD_GRAYSCALE)
image_float32 = np.float32(image)


dft = cv2.dft(image_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

shape = image.shape
H = filter_1(shape, T, a, b)
dft_noised_shift = cv2.mulSpectrums(dft_shift, H, flags=0)
dft_noised = np.fft.ifftshift(dft_noised_shift)

image_noised = cv2.idft(dft_noised)
image_noised = cv2.magnitude(image_noised[:, :, 0], image_noised[:, :, 1])

# 归一化输出
cv2.normalize(image_noised, image_noised, 0, 255, cv2.NORM_MINMAX)
image_noised = np.uint8(image_noised)

cv2.imshow('original image', image)
cv2.imshow('noised image', image_noised)
cv2.waitKey(0)  # 参数为0表示无限等待，任何键按下后退出
cv2.destroyAllWindows()  # 关闭所有窗口


