import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        self.model = load_model(os.path.join("artifacts", "training", "model.h5"))

    def predict(self):
        test_image = image.load_img(self.filename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0

        result = np.argmax(self.model.predict(test_image), axis=1)
        class_names = ["Cyst", "Normal", "Stone", "Tumor"]
        
        # Return the string directly to match the frontend expectation
        return class_names[result[0]]