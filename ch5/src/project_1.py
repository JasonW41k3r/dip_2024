import cv2
import numpy as np
import random

N = 7 # 滤波核的维数，只能为奇数
pa = pb = 0.2 # 椒盐噪声出现的概率

# 读取图像并转化为numpy数组
image_a = cv2.imread('a.jpeg', cv2.IMREAD_GRAYSCALE)
noised_matrix = image_a.copy()
# print(a_matrix.shape)



# 添加椒盐噪声
for i in range(noised_matrix.shape[0]):
    for j in range(noised_matrix.shape[1]):
        random_value = random.uniform(0, 1)
        if pa + pb > random_value > pa:
            noised_matrix[i][j] = 255
        elif random_value < pa:
            noised_matrix[i][j] = 0

temp_matrix = np.pad(noised_matrix, pad_width=((N//2, N//2), (N//2, N//2)), mode='constant', constant_values=255) # 补零填充
mf_matrix = np.zeros((noised_matrix.shape[0], noised_matrix.shape[1]))
# print(f"{a_matrix.shape, image_mf.shape}")

# 中值滤波操作
for i in range(N//2, temp_matrix.shape[0] - N//2):
    for j in range(N//2, temp_matrix.shape[1] - N//2):
        filter_kernel = temp_matrix[i-(N//2):i+(N//2)+1, j-(N//2):j+(N//2)+1]
        # print(filter_kernel)
        new_value = np.median(filter_kernel)
        # print(new_value)
        mf_matrix[i - N//2][j - N//2] = new_value
mf_matrix = np.uint8(mf_matrix) # 将中值滤波后的图像转换为 uint8 类型

# 计算处理后图像和原图像的均方差和psnr，用于评估处理效果
mse = np.mean((mf_matrix - image_a) ** 2)
psnr = cv2.PSNR(image_a, mf_matrix)

cv2.imwrite('./img/a/Before_filter.jpeg', noised_matrix)
cv2.imwrite('./img/a/After_filter.jpeg', mf_matrix)
print(f"N: {N}\nmse error: {mse}\npsnr: {psnr}")



cv2.waitKey(0)  # 参数为0表示无限等待，任何键按下后退出
cv2.destroyAllWindows()  # 关闭所有窗口