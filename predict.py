import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model("malaria_model.h5")

def predict_image(path):

    img = cv2.imread(path)
    img = cv2.resize(img,(128,128))
    img = img/255.0
    img = np.reshape(img,(1,128,128,3))

    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        return "Uninfected"
    else:
        return "Parasitized"