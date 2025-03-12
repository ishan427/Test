import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import kagglehub
import shutil

try:
    from google.colab import drive
    IS_COLAB = True
except:
    IS_COLAB = False
    pass

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Mount Google Drive
try:
    drive.mount('/content/drive')
    DRIVE_PATH = '/content/drive/MyDrive/food_spoilage_detection'
    os.makedirs(DRIVE_PATH, exist_ok=True)
except:
    DRIVE_PATH = 'food_spoilage_detection'
    os.makedirs(DRIVE_PATH, exist_ok=True)
    print("Not using Colab or drive already mounted")

# Set the path to the dataset
DATA_PATH = os.path.join(DRIVE_PATH, 'dataset')

if not os.path.exists(DATA_PATH):
    print(f"Dataset not found at {DATA_PATH}")
    DATA_PATH = DRIVE_PATH

# Define constants
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 2

def prepare_data():
    print("Preparing data...")

    # First, try with capitalized folder names
    train_dir = os.path.join(DATA_PATH, 'Train')
    test_dir = os.path.join(DATA_PATH, 'Test')

    # If that doesn't work, try with lowercase folder names
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        train_dir = os.path.join(DATA_PATH, 'train')
        test_dir = os.path.join(DATA_PATH, 'test')

    # If neither exists, try to look for the dataset structure
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print(f"Expected directory structure not found at {DATA_PATH}")

        # Check if dataset is in current directory
        if os.path.exists(os.path.join(os.getcwd(), 'train')) and os.path.exists(os.path.join(os.getcwd(), 'test')):
            train_dir = os.path.join(os.getcwd(), 'train')
            test_dir = os.path.join(os.getcwd(), 'test')
            print(f"Using train/test directories in current working directory")
        else:
            print("Could not find train/test directories. Please make sure the dataset is correctly structured.")
            return None, None

    print(f"Using train directory: {train_dir}")
    print(f"Using test directory: {test_dir}")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, test_generator

def build_model(num_classes=NUM_CLASSES):
    print(f"Building model for {num_classes} classes...")
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(model, train_generator, test_generator):
    print("Training model...")
    # Create an instance of ModelCheckpoint instead of passing the class
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(DRIVE_PATH, 'best_model.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5)

    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=test_generator,
        callbacks=[checkpoint, early_stopping, lr_reduction]  # Pass instances, not classes
    )
    return history


def main():
    train_generator, test_generator = prepare_data()
    if train_generator is None or test_generator is None:
        print("Data preparation failed. Please check the dataset structure.")
        return

    # Get the number of classes from the data
    num_classes = len(train_generator.class_indices)
    print(f"Detected {num_classes} classes in the dataset")
    print(f"Class mapping: {train_generator.class_indices}")

    # Build the model with the correct number of classes
    model = build_model(num_classes)

    # Train the model
    history = train_model(model, train_generator, test_generator)

    # Evaluate the model
    print("Evaluating model on test data...")
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    # Save the final model
    model.save(os.path.join(DRIVE_PATH, 'final_model.h5'))
    print(f"Model saved to {os.path.join(DRIVE_PATH, 'final_model.h5')}")

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
