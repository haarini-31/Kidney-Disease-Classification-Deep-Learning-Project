import os
import numpy as np
from pathlib import Path

class PredictionPipeline:
    _model = None  # Class-level model cache

    def __init__(self):
        # Resolve model path dynamically relative to standard layout
        self.model_path = Path(__file__).resolve().parent.parent / "artifacts" / "training" / "model.h5"
        # Fallback if running directly inside ai_service/ folder (or Docker container)
        if not self.model_path.exists():
            self.model_path = Path(__file__).resolve().parent / "model.h5"

        self.class_mapping = {
            0: "Cyst",
            1: "Normal",
            2: "Stone",
            3: "Tumor"
        }

    @classmethod
    def _get_model(cls, model_path):
        if cls._model is None:
            print("LzLoad: Setting CPU thread limits...")
            import tensorflow as tf
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
            tf.config.set_visible_devices([], 'GPU')

            from tensorflow.keras.models import load_model

            # Reconstruct model from part files if it does not exist
            if not model_path.exists():
                # Look for parts in standard parent location first
                parent_parts = Path(__file__).resolve().parent.parent / "artifacts" / "training"
                part_files = sorted(parent_parts.glob("model.h5.part-*"))
                
                # Fallback: look in local directory (for Docker setup)
                if not part_files:
                    local_parts = Path(__file__).resolve().parent
                    part_files = sorted(local_parts.glob("model.h5.part-*"))

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

    def predict(self, filename_or_bytes):
        from tensorflow.keras.preprocessing import image
        model = self._get_model(self.model_path)

        # Can accept bytes stream or filename
        if isinstance(filename_or_bytes, bytes):
            import io
            from PIL import Image
            img = Image.open(io.BytesIO(filename_or_bytes))
            # Ensure RGB conversion
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((224, 224))
            test_image = image.img_to_array(img)
        else:
            test_image = image.load_img(filename_or_bytes, target_size=(224, 224))
            test_image = image.img_to_array(test_image)

        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0

        predictions = model.predict(test_image)[0]
        result_idx = int(np.argmax(predictions))
        confidence = float(predictions[result_idx])

        return self.class_mapping[result_idx], confidence
