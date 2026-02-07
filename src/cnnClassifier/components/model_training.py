import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.data_ingestion_config import TrainingConfig # Ensure this path is correct

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None # Initialize as None

    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
        
    def train_valid_generator(self):
        img_size = tuple(map(int, self.config.params_image_size[:2]))

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=str(self.config.training_data),
            target_size=img_size,
            batch_size=self.config.params_batch_size,
            class_mode='categorical',
            shuffle=False,
            interpolation="bilinear"
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                rotation_range=40,
                zoom_range=0.2,
                shear_range=0.2
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=str(self.config.training_data),
            target_size=img_size,
            batch_size=self.config.params_batch_size,
            class_mode='categorical',
            shuffle=True,
            interpolation="bilinear"
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self):
        if self.model is None:
            raise ValueError("Model not loaded. Call get_base_model() first.")

        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        # Safety check for small datasets
        if self.steps_per_epoch == 0: self.steps_per_epoch = 1
        if self.validation_steps == 0: self.validation_steps = 1

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            validation_data=self.valid_generator,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            verbose=1 
        ) 

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )