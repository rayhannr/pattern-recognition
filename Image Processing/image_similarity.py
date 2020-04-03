import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

images = []
preprocessed_images = []
for i in range(0,5):
    image = Image.open('image{}.jpg'.format(i+1))
    images.append(image)
    
    image = image.rotate(-45)
    gs = image.convert(mode = 'L')
    
    width = 303.75
    height = 379.6875
    
    gs.thumbnail((width, height))
    gs = gs.crop((width / 4, height / 4, 3 * width / 4, 3 * height / 4))
    gs = gs.filter(ImageFilter.GaussianBlur(radius=0.75))
    gs.save('image_gs{}.jpg'.format(i+1))
    preprocessed_images.append(gs)

def show_image(img):
    fig = plt.figure(figsize=(18,18))
    imgs = ("Image 1", img[0]), ("Image 2", img[1]), ("Image 3", img[2]), ("Image 4", img[3]), ("Image 5",img[4])

    for (i, (name, image)) in enumerate(imgs):
        axis = fig.add_subplot(1, 5, i + 1)
        axis.set_title(name)
        plt.imshow(image, cmap = plt.cm.gray)
        plt.axis("off")

    plt.show()

show_image(images)
show_image(preprocessed_images)

image_pixel = []
for i in range(0,5):
    pixel = np.asarray(preprocessed_images[i])
    image_pixel.append(pixel)

def multiply_column_sum(img1, img2):
    return np.sum(img1.astype('float') * img2.astype('float'))


def quadratic_sum(img):
    return np.sqrt(np.sum(np.square(img.astype('float'))))

def cosine_similarity(img1, img2):
    return multiply_column_sum(img1, img2) / (quadratic_sum(img1) * quadratic_sum(img2))

def euclidean_distance(img1, img2):
    return np.sqrt(np.sum(np.square(img1.astype("float") - img2.astype("float"))))

def compare_image(img1, img2, img1_px, img2_px):
    cossim = cosine_similarity(img1_px, img2_px)
    ecl = euclidean_distance(img1_px, img2_px)
    fig = plt.figure(figsize=(5,5))
    plt.suptitle("Cosine: %.2f, Euclidean: %.2f" % (cossim, ecl))    
    
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(img1, cmap = plt.cm.gray)
    plt.axis("off")

    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(img2, cmap = plt.cm.gray)
    plt.axis("off")
    
    plt.show()
    
for i in range(0,5):
    for j in range(i+1, 5):
        compare_image(preprocessed_images[i], preprocessed_images[j], image_pixel[i], image_pixel[j])