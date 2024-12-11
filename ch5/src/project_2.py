import cv2
import numpy as np
import math

def motion_blur_filter(shape, T=1, a=0.1, b=0.1):
    """
    根据公式5.77构建运动模糊滤波器H(u,v)
    shape: 图像尺寸 (rows, cols)
    T: 模糊曝光时间(这里为1)
    a,b: 控制模糊方向和长度的参数
    """
    M, N = shape
    # 频率坐标u,v（使用numpy的fftshift风格从 -N/2~N/2 来定义）
    u = np.arange(M)
    v = np.arange(N)
    u = u - M//2
    v = v - N//2
    U, V = np.meshgrid(u, v, indexing='ij')

    # 避免分母为0的情况
    denom = (U*a + V*b)
    H = np.zeros((M, N), dtype=complex)

    # 对于非零点
    nonzero_mask = denom != 0
    H[nonzero_mask] = (T / (np.pi * denom[nonzero_mask])) * \
        np.sin(np.pi * denom[nonzero_mask]) * np.exp(-1j * np.pi * denom[nonzero_mask])

    # 求极限
    H[~nonzero_mask] = T

    return H

def wiener_filter(G, H, K=0.01):
    """
    维纳滤波器实现：
    F_hat(u,v) = H*(u,v) / [|H(u,v)|^2 + K] * G(u,v)
    G(u,v) 是受损图像的频谱（模糊+噪声）
    H(u,v) 是已知的运动模糊滤波器
    K 是平衡参数
    """
    H_conj = np.conjugate(H)
    H_abs2 = np.abs(H)**2
    F_hat = (H_conj / (H_abs2 + K)) * G
    return F_hat

def add_gaussian_noise(image, mean=0, sigma=10):
    """
    在图像上添加高斯噪声
    mean: 均值
    sigma: 标准差
    """
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def dft2(image):
    return np.fft.fftshift(np.fft.fft2(image))

def idft2(freq_image):
    return np.real(np.fft.ifft2(np.fft.ifftshift(freq_image)))

# ---------------- 主流程 ----------------
if __name__ == "__main__":
    # 读取灰度图像
    img = cv2.imread('b.jpeg', cv2.IMREAD_GRAYSCALE)

    # 将图像转为浮点型进行后续处理
    img_float = img.astype(np.float64)

    # 构建运动模糊滤波器 H(u,v)
    H = motion_blur_filter(img_float.shape, T=1, a=0.1, b=0.1)

    # 对图像进行模糊处理（频域操作）
    F = dft2(img_float)
    G = F * H  # 模糊后的频域表示
    blurred = idft2(G)
    blurred = np.clip(blurred, 0, 255).astype(np.uint8)

    # 添加高斯噪声
    noisy_blurred = add_gaussian_noise(blurred, mean=0, sigma=10)

    # 转换回频域
    G_noisy = dft2(noisy_blurred.astype(np.float64))

    # 4. 维纳滤波恢复图像
    K = 0.01
    F_hat = wiener_filter(G_noisy, H, K=K)
    restored = idft2(F_hat)
    restored = np.clip(restored, 0, 255).astype(np.uint8)

    # 保存结果
    cv2.imwrite('./img/b/blurred_image.jpeg', blurred)
    cv2.imwrite('./img/b/noisy_blurred_image.jpeg', noisy_blurred)
    cv2.imwrite('./img/b/restored_image.jpeg', restored)


