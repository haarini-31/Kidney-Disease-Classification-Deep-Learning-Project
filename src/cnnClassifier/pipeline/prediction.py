import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from pathlib import Path
import os

class PredictionPipeline:
    def __init__(self):
        # Resolve path dynamically relative to this file
        self.project_root = Path(__file__).resolve().parent.parent.parent.parent
        model_path = self.project_root / "artifacts" / "training" / "model.h5"
        
        # Load the model only once when the pipeline is instantiated
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        self.model = load_model(str(model_path))
        
        # Explicit class mapping (based on flow_from_directory alphanumeric sorting)
        self.class_mapping = {
            0: "Cyst",
            1: "Normal",
            2: "Stone",
            3: "Tumor"
        }

    def predict(self, filename):
        test_image = image.load_img(filename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        
        # Normalization exactly matches the training rescale=1./255
        test_image = test_image / 255.0

        predictions = self.model.predict(test_image)[0]
        result_idx = int(np.argmax(predictions))
        confidence = float(predictions[result_idx])
        
        predicted_class = self.class_mapping[result_idx]
        
        return predicted_class, confidence