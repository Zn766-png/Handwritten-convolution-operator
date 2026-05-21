# Handwritten-convolution-operator

CV

## Experimental Objectives:
Master dimension calculation and understand the influence of Padding (filling) and Stride (step size) in convolution operations on the output size.
Implement 2D convolution without using deep learning frameworks, using only Numpy slicing and loops to implement the forward propagation of conv2d.
Apply the Sobel kernel to observe edge feature extraction.
Implement Max Pooling and understand the downsampling process.

## Experimental data: 
Use the built-in classic image "ascent" (or "camera") from scipy.datasets (or skimage). Preprocessing: Resize to (128, 128) to accelerate the calculation, convert to float32 and normalize to the range [0, 1].

## Experimental content:
Use np.pad(img, ((P, P), (P, P)), 'constant') to pad the original image with zeros.
Implement the 2D convolution function, my_conv2d(img, kernel, stride=1, padding=0).
Input: img (H,W), kernel (K,K).
Output size (must be calculated using formulas and asserted): H_out=⌊H+2P−K/S⌋+1, W_out=⌊W+2P−K/S⌋+1
Calculation method: For each output position (i, j): region = padded[i*S : i*S+K, j*S : j*S+K]; out[i,j] = np.sum(region * kernel).
Edge detection: Define Sobel X (vertical edge) and Sobel Y (horizontal edge) convolution kernels (3x3), respectively call my_conv2d, and output two feature maps. Use matplotlib to plot: original image vs vertical edge map vs horizontal edge map.
Implement the Max Pooling (maximum pooling) function my_maxpool2d(img, kernel_size=2, stride=2). The logic is the same as convolution, but no padding (default), and no weights. Window operation is changed to find the maximum value: out[i, j] = np.max(region).

# 实验 手写卷积算子

## 实验目标：
掌握维度计算，理解卷积操作中 Padding（填充）、Stride（步长）对输出尺寸的影响。
实现2D卷积，不调用深度学习框架，仅用 Numpy 切片和循环实现 conv2d 前向传播。
应用 Sobel 核观察边缘特征提取。
实现Max Pooling，理解下采样过程。

## 实验数据：
使用 scipy.datasets (或 skimage) 内置经典图像 ascent (或 camera)。预处理：resize 至 (128, 128)以加速计算，转 float32并归一化至 [0, 1] 区间。

## 实验内容：
使用 np.pad(img, ((P, P), (P, P)), 'constant') 对原图进行零填充。
实现 2D 卷积函数，my_conv2d(img, kernel, stride=1, padding=0)。
输入：img (H,W), kernel (K,K)。
输出尺寸（必须用公式计算并 assert）：
H_out=⌊H+2P−K/S⌋+1, W_out=⌊W+2P−K/S⌋+1
计算方式：对每个输出位置 (i, j)：region = padded[i*S : i*S+K, j*S : j*S+K]；out[i,j] = np.sum(region * kernel)。
边缘检测，定义 Sobel X (垂直边缘) 和 Sobel Y (水平边缘) 卷积核 (3x3)，分别调用 my_conv2d，输出两张特征图。使用 matplotlib 绘制：原图 vs 垂直边缘图 vs 水平边缘图。
实现 Max Pooling (最大池化)函数 my_maxpool2d(img, kernel_size=2, stride=2)。逻辑同卷积，但不需要 padding（默认），不需要权重。窗口操作改为求最大值：out[i, j] = np.max(region)。
