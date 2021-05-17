
# Import libraries
import tensorflow as tf
from tensorflow import keras
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as imd
from skimage import transform
from skimage.color import rgb2gray

# Directory routes for datasets
main_dir = 'datasets/belgian/'
train_data_dir = os.path.join(main_dir, 'Training')
test_data_dir = os.path.join(main_dir, 'Testing')

# Function for loading belgian train and test datasets
def load_data(data_directory):
    dirs = [d for d in os.listdir(data_directory)
           if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in dirs:
        label_dir = os.path.join(data_directory, d)
        file_names = [os.path.join(label_dir, f)
                     for f in os.listdir(label_dir)
                     if f.endswith('.ppm')]
        for f in file_names:
            images.append(imd.imread(f))
            labels.append(int(d))
    
    return images, labels

# Functions for showing predictions using matplotlib
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(predicted_label,
                                100*np.max(predictions_array),
                                true_label),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(62))
    plt.yticks([])
    thisplot = plt.bar(range(62), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# Pre-process dataset
train_images, train_labels = load_data(train_data_dir)
test_images, test_labels = load_data(test_data_dir)

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

train_images = [transform.resize(image, (30,30)) for image in train_images]
test_images = [transform.resize(image, (30,30)) for image in test_images]

train_images = np.array(train_images)
test_images = np.array(test_images)

train_images = rgb2gray(train_images)
test_images = rgb2gray(test_images)

# Model building
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(30, 30)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(62, activation='softmax')
])

# Model compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Model training
model.fit(train_images, train_labels, epochs=20)

# Validate model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(f'Accuracy: {test_acc}')

# Predictions with random images
predictions = model.predict(test_images)

# Generate random labels
rand_signs = random.sample(range(0, len(test_labels)), 15)
rand_signs

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(len(rand_signs)):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(rand_signs[i], predictions[rand_signs[i]], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(rand_signs[i], predictions[rand_signs[i]], test_labels)
plt.tight_layout()
plt.show()
