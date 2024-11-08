import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Đọc ảnh và chuyển thành ảnh RGB
image = cv2.imread('images/AnhVeTinh.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Chuyển đổi ảnh thành dạng vector (mỗi pixel là một điểm)
pixels = image_rgb.reshape((-1, 3))

# Áp dụng K-means
kmeans = KMeans(n_clusters=2, random_state=42)  # Giả sử có 2 cụm: nhà và không phải nhà
kmeans.fit(pixels)

# Gán nhãn cho các pixel
labels = kmeans.labels_

# Chuyển đổi nhãn thành ảnh
segmented_image = labels.reshape(image_rgb.shape[0], image_rgb.shape[1])

# Hiển thị ảnh phân cụm
plt.imshow(segmented_image, cmap='viridis')
plt.show()
