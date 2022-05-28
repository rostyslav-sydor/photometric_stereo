import numpy as np
from numpy import linalg
from PIL import Image, ImageOps
from sys import argv
from os import walk

if len(argv) == 1:
    exit("No images have been found")

images = []
for root, dirs, files in walk(argv[1]):
    sources = np.zeros(shape=(len(files),3))
    for i, file in enumerate(files):
        img = Image.open(root + '/' + file)
        imgIntensity = ImageOps.grayscale(img)
        images.append(np.asfarray(imgIntensity))
        sources[i] = file.split('.')[0].split('_')

width = images[0].shape[0]
height = images[0].shape[1]
# calculate intensities
intensities = np.zeros(shape=(width, height, len(images)))

for it, img in enumerate(images):
    for i in range(intensities.shape[0]):
        for j in range(intensities.shape[1]):
            intensities[i][j][it] = img[i][j]


# process sources
trans = np.transpose(sources)
mat = linalg.inv(np.matmul(trans, sources))

# calculate normals
normals = np.zeros(shape=(width, height, 3))

for i in range(width):
    for j in range(height):
        norm = np.matmul(mat, np.matmul(trans,intensities[i][j]))
        norm /= np.linalg.norm(norm)
        # norm[1], norm[2] = norm[2], norm[1]
        normals[i][j] = 0.5*norm + 0.5

        # print(norm)

newimg = Image.fromarray((normals * 255).astype(np.uint8)).save('norm.jpg')
