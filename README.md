#Building a COVID-19 Face Mask Detection System using Simple CNN

Introduction:
In this blog, we will explore the development of a COVID-19 face mask detection system using deep learning techniques. The goal is to build a model that can classify whether a person is wearing a face mask or not. We will utilize the power of convolutional neural networks (CNNs) and the Keras library to achieve this task. The dataset used for training and testing consists of images of individuals with and without face masks.

Understanding the Code:

1. Importing Libraries:
To begin, we import the necessary libraries for our project. These libraries include numpy, pandas, cv2 (OpenCV), and ImageDataGenerator from the TensorFlow library. These libraries will help us with data manipulation, image processing, and data augmentation.

```python
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```

2. Data Preprocessing:
Before training our model, we need to preprocess our data. This involves resizing the images, applying data augmentation techniques, and normalizing pixel values. We use the `ImageDataGenerator` class from Keras to perform data augmentation on the training data. Data augmentation techniques like rotation, shifting, shearing, zooming, and flipping are applied to enhance the model's ability to generalize. The test data is only rescaled without augmentation.

```python
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)
```

3. Data Directory:
We need to specify the paths to the directories containing our training, testing, and validation images. These directories contain subdirectories for the different classes (Mask and Non Mask).

```python
train_dir = '/kaggle/input/covid-face-mask-detection-dataset/New Masks Dataset/Train'
test_dir = '/kaggle/input/covid-face-mask-detection-dataset/New Masks Dataset/Test'
val_dir = '/kaggle/input/covid-face-mask-detection-dataset/New Masks Dataset/Validation'
```

4. Data Generators:
Now, we create data generators using the `flow_from_directory` method. These generators will load and preprocess the images on the fly during training and evaluation. The generator for the training data uses the previously defined augmentation settings.

```python
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_generator = test_datagen.flow_from_directory(
    val_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)
```

5. Model Architecture:
Now it's time to define the architecture of our model. We use the `Sequential` class from Keras to build our model. It consists of several convolutional layers with max-pooling, followed by fully connected layers. The final output layer uses the sigmoid activation function to predict the probability of a person wearing a mask.

```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
```

6. Model Compilation and Training:
After defining the model, we compile it by specifying the optimizer, loss function, and evaluation metric. Then, we train the model using the `fit` method. We pass in the training generator, number of training steps per epoch, validation data, validation steps, number of epochs, and the verbosity level for progress updates.

```python
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=10,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    verbose=1
)
```

7. Testing and Visualization:
Finally, we test our trained model on new images and visualize the results. We define a list of image paths for which we want to make predictions. For each image, we read it using OpenCV, preprocess it, and store it in an array. The model's `predict` method is then used to obtain the predicted probabilities. Based on the probability, we determine whether the person is wearing a mask or not. We visualize the results by drawing bounding boxes around the faces and displaying the label and confidence.

```python
image_paths = ['/kaggle/input/covid-face-mask-detection-dataset/New Masks Dataset/Validation/Mask/1701.jpg', '/kaggle/input/covid-face-mask-detection-dataset/New Masks Dataset/Validation/Non Mask/real_00007.jpg']

images = []
for image_path in image_paths:
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = cv2.resize(image, (224, 224))  # Resize to match model input shape
    image = image / 255.0  # Normalize pixel values
    images.append(image)

images = np.array(images)

predictions = model.predict(images)

for i, prediction in enumerate(predictions):
    label = "Mask" if prediction < 0.5 else "No Mask"
    confidence = 1 - prediction[0] if label == "Mask" else prediction[0]

    # Visualize the results
    image = cv2.imread(image_paths[i])

    if label == "Mask":
        color = (0, 255, 0)  # Green color for Mask
    else:
        color = (0, 0, 255)  # Red color for No Mask

    cv2.rectangle(image, (0, 0), (image.shape[1], image.shape[0]), color, 2)

    text = f"{label} {confidence:.2f}"
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Image {i+1}: {label} {confidence:.2f}")
    plt.axis('off')
    plt.show()
```

Conclusion:
In this blog post, we have covered the complete process of building a COVID-19 face mask detection system using deep learning. We started with data preprocessing and augmentation using the `ImageDataGenerator

class from Keras. Then, we designed and trained a convolutional neural network model using the Sequential class from Keras. Finally, we used the trained model to make predictions on new images and visualize the results.

By understanding and implementing the code, we have developed a system that can effectively classify whether a person is wearing a face mask or not. This system can have various applications, including monitoring compliance with face mask regulations in public spaces, enhancing safety measures in healthcare facilities, and assisting in automated screening processes.

With further refinement and training on larger datasets, this model can potentially be deployed in real-world scenarios to help combat the spread of COVID-19. It is important to note that the success and accuracy of the model depend on the quality and diversity of the training data, as well as the chosen model architecture and hyperparameters.

Deep learning-based solutions like this face mask detection system showcase the potential of AI and computer vision in addressing public health challenges. By leveraging such technology, we can contribute to creating safer environments and minimizing the risk of virus transmission.
