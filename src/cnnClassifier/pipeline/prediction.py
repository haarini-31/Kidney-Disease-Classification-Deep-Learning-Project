import numpy as np
from pathlib import Path
import os

class PredictionPipeline:
    _model = None  # Class-level cache for the loaded model

    def __init__(self):
        # Resolve path dynamically relative to this file, but do NOT import tensorflow or load model here
        self.project_root = Path(__file__).resolve().parent.parent.parent.parent
        self.model_path = self.project_root / "artifacts" / "training" / "model.h5"
        
        # Explicit class mapping (based on flow_from_directory alphanumeric sorting)
        self.class_mapping = {
            0: "Cyst",
            1: "Normal",
            2: "Stone",
            3: "Tumor"
        }

    @classmethod
    def _get_model(cls, model_path):
        if cls._model is None:
            print("LzLoad: Setting conservative CPU thread limits for TensorFlow...")
            # Configure TensorFlow thread limits and options before loading
            import tensorflow as tf
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
            tf.config.set_visible_devices([], 'GPU')
            
            from tensorflow.keras.models import load_model
            
            # Reconstruct model from part files if it does not exist
            if not model_path.exists():
                part_files = sorted(model_path.parent.glob("model.h5.part-*"))
                if part_files:
                    print(f"LzLoad: Reconstructing model from parts: {part_files}")
                    os.makedirs(model_path.parent, exist_ok=True)
                    with open(model_path, 'wb') as outfile:
                        for part in part_files:
                            with open(part, 'rb') as infile:
                                outfile.write(infile.read())

            if not model_path.exists():
                raise FileNotFoundError(f"Model not found at {model_path}")
            
            print(f"LzLoad: Loading Keras VGG16 model from {model_path}...")
            cls._model = load_model(str(model_path))
            print("LzLoad: Model loaded successfully.")
        return cls._model

    def predict(self, filename):
        # Defer TensorFlow image preprocessors import to request runtime
        from tensorflow.keras.preprocessing import image
        
        # Retrieve cached model lazily
        model = self._get_model(self.model_path)
        
        test_image = image.load_img(filename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        
        # Normalization exactly matches the training rescale=1./255
        test_image = test_image / 255.0

        predictions = model.predict(test_image)[0]
        result_idx = int(np.argmax(predictions))
        confidence = float(predictions[result_idx])
        
        predicted_class = self.class_mapping[result_idx]
        
        return predicted_class, confidence