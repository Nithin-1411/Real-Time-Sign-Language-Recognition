from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
# from keras.preprocessing.image import ImageDataGenerator
# from keras.preprocessing.image import ImageDataGenerator
# import TensorFlow.keras.preprocessing.image.ImageDataGenerator
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import keras
# Initializing the LeNet-5 CNN
lenet_classifier = Sequential()

# Layer 1: Convolutional + Average Pooling
lenet_classifier.add(Conv2D(6, (5, 5), activation='tanh', input_shape=(64, 64, 3)))
lenet_classifier.add(AveragePooling2D((2, 2)))

# Layer 2: Convolutional + Average Pooling
lenet_classifier.add(Conv2D(16, (5, 5), activation='tanh'))
lenet_classifier.add(AveragePooling2D((2, 2)))

# Layer 3: Flatten
lenet_classifier.add(Flatten())

# Layer 4: Fully Connected
lenet_classifier.add(Dense(120, activation='tanh'))

# Layer 5: Fully Connected
lenet_classifier.add(Dense(84, activation='tanh'))

# Output Layer: Softmax for 26 classes (A-Z)
lenet_classifier.add(Dense(26, activation='softmax'))

# Compile the model
lenet_classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data Augmentation and Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    
    'C:/Users/G NITHIN/Documents/sft_project_main/Simple-Sign-Language-Detector-master/Simple-Sign-Language-Detector-master/mydata/training_set',  # Update with your training data directory
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    'C:/Users/G NITHIN/Documents/sft_project_main/Simple-Sign-Language-Detector-master/Simple-Sign-Language-Detector-master/mydata/test_set',     # Update with your test data directory
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

# Train the model
model = lenet_classifier.fit(
    training_set,
    steps_per_epoch=800,
    epochs=25,
    validation_data=test_set,
    validation_steps=6500 
)
# ... (Your existing code for model definition, training, and plotting)

# Save the model
lenet_classifier.save('lenet_sign_language_model1.keras')  # Save as an HDF5 file
lenet_classifier.save('C:/Users/G NITHIN/Documents/sft_project_main/Simple-Sign-Language-Detector-master/Simple-Sign-Language-Detector-master/lenet_sign_language_model1.keras')
print("Model saved successfully!")
# model.save('my_model.keras')
# keras.saving.save_model(model, 'my_model1.keras')

# Plotting
plt.plot(model.history['accuracy'])
plt.plot(model.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'], loc='upper left')
plt.show()

plt.plot(model.history['loss'])
# plt.plot(model.history['val 'Validation'], loc='upper left')
# plt.plot(model.history['val_loss'], label='Validation Loss', loc='upper left')

# plt.show()
plt.plot(model.history['accuracy'], label='Training Accuracy')
plt.plot(model.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# Corrected line for plotting validation loss
plt.plot(model.history['loss'], label='Training Loss')
plt.plot(model.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()
