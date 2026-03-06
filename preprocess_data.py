import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
parasitized_path =r"C:\Malaria Detection Using Deep Learning\dataset\cell_images\parasitized" 
uninfected_path = r"C:\Malaria Detection Using Deep Learning\dataset\cell_images\uninfected"

data = []
labels = []

# Read Parasitized images
for img in os.listdir(parasitized_path):

    path = os.path.join(parasitized_path, img)
    image = cv2.imread(path)

    if image is None:
        continue

    image = cv2.resize(image, (128,128))
    data.append(image)
    labels.append(0)

# Read Uninfected images
for img in os.listdir(uninfected_path):

    path = os.path.join(uninfected_path, img)
    image = cv2.imread(path)

    if image is None:
        continue

    image = cv2.resize(image, (128,128))
    data.append(image)
    labels.append(1)

data = np.array(data)
labels = np.array(labels)

print("Total Images Loaded:", len(data))
print("Total Labels:", len(labels))

print("Total Images:", len(data))


X_train, X_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.2,
    random_state=42
)

print("Training images:", len(X_train))
print("Testing images:", len(X_test))