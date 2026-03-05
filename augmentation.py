from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# load image
img = Image.open("sample.jpg")   # put any image path here
img = np.array(img)

img = img.reshape((1,) + img.shape)

# augmentation generator
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# generate augmented images
i = 0
for batch in datagen.flow(img, batch_size=1):
    
    plt.imshow(batch[0].astype('uint8'))
    plt.axis('off')
    plt.show()
    
    i += 1
    if i > 5:
        break