# 编程作业
## 编程作业#1
- 下载图(a)，加入椒盐噪声，要求 \( P_a = P_b = 0.2 \)，将得到的结果应用中值滤波，然后比较并解释与5.10(b)的主要区别。
### 提供的图片：
- 图(a)
- 图5.10(b)
---
## 编程作业#2
- 实现如公式5.77所示的滤波器，使用 \( T = 1 \) 在 \( +45^\circ \) 方向上模糊图(b)，如5.26(b)所示。
- 然后向模糊图像添加均值为0、方差为10像素的高斯噪声。
- 使用公式5.85所示的参数的维纳滤波器恢复图像。
### 公式：
- 滤波器公式（5.77）：
  \[
  H(u,v) = \frac{\sin(\pi (ua + vb))}{\pi (ua + vb)} e^{-j2\pi (ua+vb)}
  \]
- 维纳滤波器公式（5.85）：
  \[
  \hat{F}(u,v) = \frac{1}{H(u,v)} \frac{|H(u,v)|^2}{|H(u,v)|^2 + K} G(u,v)
  \]
### 提供的图片：
- 图(b)
- 图5.26(b)