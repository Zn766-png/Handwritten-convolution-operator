# 课程作业：10计算机视觉(一)-手写卷积算子
# 作者：Zn766-png
# 备注：数据来源为scipy.datasets的内置经典图像ascent;
#      预处理为resize至(128, 128)以加速计算,转float32并归一化至[0, 1]区间。

# ===========实验内容0.加载数据及预处理===========
# (1) 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
from scipy import datasets
from scipy.ndimage import zoom

# (2) 加载数据
img = datasets.ascent()  # 原始图像 (512, 512)

# (3) 预处理
img_resized = zoom(img, (128/img.shape[0], 128/img.shape[1]))   # resize至(128,128)以加速计算
img_norm = img_resized.astype(np.float32) / 255.0   # 转换为float32并归一化到[0, 1]

# ===================实验内容1.使用np.pad(img,((P,P),(P,P)),'constant')对原图进行零填充===================
P = 1  # 填充宽度
padded_img = np.pad(img_norm, ((P, P), (P, P)), mode='constant')
print(f"原图尺寸{img_norm.shape}")
print(f"零填充后尺寸{padded_img.shape}")

# ===================实验内容2.实现2D卷积函数,my_conv2d(img, kernel, stride=1, padding=0)===================
# (a) 输入img(H,W),kernel(K,K):img是输入图像 (H, W),kernel是积核 (K, K),stride是步长,padding是填充宽度
def my_conv2d(img, kernel, stride=1, padding=0):   # 2D卷积函数
    assert img.ndim == 2, "Input image must be 2D"  # 断言,防御性编程,确保程序健壮性,确保输入是二维
    assert kernel.ndim == 2, "Kernel must be 2D"
    H, W = img.shape
    K = kernel.shape[0]  # 假设正方形卷积核
    S = stride
    P = padding

# (b) 输出尺寸(必须用公式计算并assert)H_out=|(H+2P−K)/S|+1,W_out=|(W+2P−K)/S|+1断言输出尺寸为正
    H_out = (H + 2 * P - K) // S + 1
    W_out = (W + 2 * P - K) // S + 1
    assert H_out > 0 and W_out > 0, f"Invalid output size: ({H_out}, {W_out})"

    if P > 0:
        img_padded = np.pad(img, ((P, P), (P, P)), mode='constant')
    else:
        img_padded = img
    out = np.zeros((H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):

# (c) 计算方式对每个输出位置(i,j):region=padded[i*S:i*S+K,j*S:j*S+K];out[i,j]=np.sum(region*kernel)
            h_start = i * S     # 提取当前区域
            h_end = h_start + K
            w_start = j * S
            w_end = w_start + K
            region = img_padded[h_start:h_end, w_start:w_end]
            out[i, j] = np.sum(region * kernel)     # 计算点积并求和
    return out   # 卷积结果为(H_out, W_out)

# ===================实验内容3.边缘检测===================
# (1) 定义Sobel X(垂直边缘)和Sobel Y(水平边缘)卷积核(3x3)
sobel_x = np.array([[-1, 0, 1],    # 垂直边缘
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)
sobel_y = np.array([[-1, -2, -1],    # 水平边缘
                    [0, 0, 0],
                    [1, 2, 1]], dtype=np.float32)

# (2) 分别调用my_conv2d
edge_x = my_conv2d(img_norm, sobel_x, stride=1, padding=1)
edge_y = my_conv2d(img_norm, sobel_y, stride=1, padding=1)

# (3) 输出两张特征图
print(f"垂直边缘图的形状{edge_x.shape}")
print(f"水平边缘图的形状{edge_y.shape}")    # 输出两张特征图(即两个独立的张量/数组),分别表示水平梯度和垂直梯度,深度学习或图像处理里就是这样进行分量的分离计算的

# (4) 使用matplotlib绘制原图、垂直边缘图、水平边缘图
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)    # 原图
plt.imshow(img_norm, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 3, 2)    # 垂直边缘图
plt.imshow(edge_x, cmap='gray')
plt.title('Vertical Edges (Sobel X)')
plt.axis('off')
plt.subplot(1, 3, 3)    # 水平边缘图
plt.imshow(edge_y, cmap='gray')
plt.title('Horizontal Edges (Sobel Y)')
plt.axis('off')
plt.tight_layout()
plt.show()

# ===================实验内容4.实现Max Pooling(最大池化)函数my_maxpool2d(img, kernel_size=2, stride=2)===================
# (1) 逻辑同卷积,但不需要padding(默认),不需要权重
def my_maxpool2d(img, kernel_size=2, stride=2):   # Max Pooling(最大池化)函数
    H, W = img.shape    # img为输入特征图(H, W),kernel_size为池化窗口大小,stride为步长
    K = kernel_size
    S = stride
    H_out = (H - K) // S + 1    # 输出尺寸
    W_out = (W - K) // S + 1
    out = np.zeros((H_out, W_out))   # 初始化输出
    for i in range(H_out):  # 开始执行最大池化
        for j in range(W_out):
            h_start = i * S
            h_end = h_start + K
            w_start = j * S
            w_end = w_start + K
            region = img[h_start:h_end, w_start:w_end]

# (2) 窗口操作改为求最大值out[i,j]=np.max(region)
            out[i, j] = np.max(region)
    return out   # 输出池化后的特征图(H_out, W_out)

# (3) 额外进行一次对my_maxpool2d函数的测试并可视化
pooled = my_maxpool2d(edge_x, kernel_size=2, stride=2)
print(f"最大池化后形状为{pooled.shape}")
plt.figure(figsize=(8, 4))    # 最大池化后的效果图
plt.subplot(1, 2, 1)
plt.imshow(edge_x, cmap='gray')
plt.title('Before Max Pooling')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(pooled, cmap='gray')
plt.title('After Max Pooling (2x2)')
plt.axis('off')
plt.tight_layout()
plt.show()