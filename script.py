import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import os

# Define paths for dataset
train_dir = "dataset/train/"
test_dir = "dataset/test/"

# Image Data Generator for training and testing
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load train and test data
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=32, class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(224, 224), batch_size=32, class_mode="categorical"
)

# Load MobileNetV2 as base model
base_model = MobileNetV2(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)

# Build the model
model = Sequential(
    [
        base_model,
        GlobalAveragePooling2D(),
        Dense(1024, activation="relu"),
        Dense(5, activation="softmax"),  # 5 output classes
    ]
)

# Freeze the base model layers
base_model.trainable = False

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Train the model
history = model.fit(train_generator, epochs=10, validation_data=test_generator)

# Save the model
model.save("wild_animal_classifier.h5")

# Load and predict on a new image
from tensorflow.keras.preprocessing import image
import numpy as np


def predict_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values

    # Predict the class probabilities
    predictions = model.predict(img_array)

    # Get the class names and their corresponding indices
    class_indices = train_generator.class_indices
    class_names = {v: k for k, v in class_indices.items()}  # Reverse mapping

    # Get the predicted class index and name
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class_index]

    # Get the confidence (accuracy) of the prediction
    confidence = predictions[0][predicted_class_index] * 100  # Convert to percentage

    # Return the predicted class name and confidence
    return predicted_class_name, confidence


# Example of usage:
# img_path = "test_images/elephant1.jpg"
img_path = "test_images/tiger1.jpg"
# img_path = "test_images/peacock1.jpg"
# img_path = "test_images/leopard1.jpg"
# img_path = "test_images/wild_boar1.jpg"
predicted_animal, accuracy = predict_image(img_path)
print(f"Predicted animal: {predicted_animal} with accuracy: {accuracy:.2f}%")
