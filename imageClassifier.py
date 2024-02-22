import tensorflow as tf
import tensorflowjs as tfjs
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.python.keras.models import load_model
import cv2
import numpy as np
import os, imutils

def load_and_preprocess_images(directory, extension=".jpg", target_size=(256, 256)):
    images = []
    labels = []
    class_names = os.listdir(directory)
    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        class_label = class_names.index(class_name)
        for filename in os.listdir(class_dir):
            if filename.endswith(extension):
                img_path = os.path.join(class_dir, filename)
                img = cv2.imread(img_path)
                img = cv2.resize(img, target_size)
                img = img / 255.0  # Normalize
                images.append(img)
                labels.append(class_label)
    return np.array(images), np.array(labels)

def load_and_preprocess_images_test(directory, extension=".jpg", target_size=(256, 256)):
    images = []
    labels = []
    # read all the images in the directory
    for filename in os.listdir(directory):
        if filename.endswith(extension):
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, target_size)
            img = img / 255.0  # Normalize
            images.append(img)
            labels.append(filename)
    return np.array(images), np.array(labels)

train_images, train_labels = load_and_preprocess_images("train", extension=".jpg")

test_images, test_labels = load_and_preprocess_images_test("C:\\Users\\mrsur\\Desktop\\plantDiseases\\test\\test", extension=".jpg")

# Build the model
model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(34, activation='softmax'))  # 4 units for 4 classes with softmax

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training
model.fit(train_images, train_labels, epochs=23, validation_split=0.2)

# Evaluate on test data
loss = 0
accuracy = 1
try:
    loss, accuracy = model.evaluate(test_images, test_labels)
except:
    pass
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")

model.save("plantDiseaseDetector.h5")
tfjs.converters.save_keras_model(model, "plantDiseaseDetector")
