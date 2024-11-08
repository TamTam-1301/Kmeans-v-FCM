import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import cv2

# Đọc ảnh và chuyển thành ảnh RGB
image = cv2.imread('images/AnhVeTinh.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Chuyển ảnh thành dạng vector
pixels = image_rgb.reshape((-1, 3))

# Chuẩn bị dữ liệu cho FCM
pixels = pixels.T

# Số cụm cần phân
n_clusters = 2

# Áp dụng FCM
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(pixels, n_clusters, m=2, error=0.005, maxiter=1000)

# Gán nhãn cho mỗi pixel theo mức độ thuộc về cụm
labels = np.argmax(u, axis=0)

# Chuyển nhãn thành ảnh phân cụm
segmented_image = labels.reshape(image_rgb.shape[0], image_rgb.shape[1])

# Hiển thị ảnh phân cụm
plt.imshow(segmented_image, cmap='viridis')
plt.show()
