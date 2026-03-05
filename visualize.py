import matplotlib.pyplot as plt
import os
from PIL import Image

path = r"C:\Malaria Detection Using Deep Learning\New Plant Diseases Dataset\New Plant Diseases Dataset(Augmented)\Dataset\train"

folders = os.listdir(path)

plt.figure(figsize=(15,15))

for i, folder in enumerate(folders[:12]):   # show first 12 classes
    folder_path = os.path.join(path, folder)

    image_name = os.listdir(folder_path)[0]
    image_path = os.path.join(folder_path, image_name)

    img = Image.open(image_path)

    plt.subplot(3,4,i+1)
    plt.imshow(img)
    plt.title(folder)
    plt.axis("off")

plt.tight_layout()
plt.show()