import numpy as np
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread("panoramic.jpeg")
img2 = cv2.imread("1.jpg")

def vectorize_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_vectorized = img.reshape((-1,3))
    img_vectorized = np.float32(img_vectorized)
    
    return img_vectorized

img1_vectorized = vectorize_image(img1)
img2_vectorized = vectorize_image(img2)

def calculate_sse(data, n_clusters, label, centroid):
    error = 0
    for i in range(n_clusters):
        error += np.sum(np.square(data[label == i] - centroid[i]))
        
    return error

from sklearn.metrics import pairwise_distances_argmin
def kmeans(data, n_clusters):
    rand = np.random.RandomState(42)
    random_point = rand.permutation(data.shape[0])[:n_clusters]
    centroid = data[random_point]
    
    while True:
        label = pairwise_distances_argmin(data, centroid)
        new_centroid = np.array([data[label == i].mean(0) if data[label == i].size != 0 else [0,0,0] for i in range(n_clusters)])
        
        if np.all(centroid == new_centroid):
            break
        centroid = new_centroid
        
    return centroid, label

def convert_color(data, label, centroid):
    new_data = np.copy(data)
    for i in range(centroid.shape[0]):
        new_data[label == i] = centroid[i].astype(int)
    
    return new_data

def visualize(original_image, segmented_image, label, centroid):
    result_image = convert_color(segmented_image, label, centroid)
    result_image = np.uint8(result_image.reshape(original_image.shape))
    
    figure_size = 15
    plt.figure(figsize=(figure_size,figure_size))
    plt.subplot(1,2,2),plt.imshow(result_image)
    plt.title('Segmented Image'), plt.xticks([]), plt.yticks([])
    plt.show()

centroid12, label12 = kmeans(img1_vectorized, 2)
visualize(img1, img1_vectorized, label12, centroid12)

centroid13, label13 = kmeans(img1_vectorized, 3)
visualize(img1, img1_vectorized, label13, centroid13)

centroid22, label22 = kmeans(img2_vectorized, 2)
visualize(img2, img2_vectorized, label22, centroid22)

centroid23, label23 = kmeans(img2_vectorized, 3)
visualize(img2, img2_vectorized, label23, centroid23)

def elbow_method(data):
    wcss = []
    for i in range(1, 11):
        cent, lab = kmeans(data, i)
        err = calculate_sse(data, i, lab, cent)
        wcss.append(err)
    plt.plot(range(1, 11), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.show()

elbow_method(img1_vectorized)
elbow_method(img2_vectorized)