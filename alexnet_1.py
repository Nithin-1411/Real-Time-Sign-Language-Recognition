import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# --- AlexNet Model Definition ---
alexnet_classifier = Sequential()

alexnet_classifier.add(Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=(227, 227, 3)))
alexnet_classifier.add(MaxPooling2D((3, 3), strides=(2, 2)))

alexnet_classifier.add(Conv2D(256, (5, 5), padding='same', activation='relu'))
alexnet_classifier.add(MaxPooling2D((3, 3), strides=(2, 2)))

alexnet_classifier.add(Conv2D(384, (3, 3), padding='same', activation='relu'))
alexnet_classifier.add(Conv2D(384, (3, 3), padding='same', activation='relu'))
alexnet_classifier.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
alexnet_classifier.add(MaxPooling2D((3, 3), strides=(2, 2)))

alexnet_classifier.add(Flatten())
alexnet_classifier.add(Dense(4096, activation='relu'))
alexnet_classifier.add(Dropout(0.5))
alexnet_classifier.add(Dense(4096, activation='relu'))
alexnet_classifier.add(Dropout(0.5))

alexnet_classifier.add(Dense(26, activation='softmax')) 

alexnet_classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- Data Augmentation and Preprocessing ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    # Update with your training data directory
    'C:/Users/G NITHIN/Documents/sft_project_main/Simple-Sign-Language-Detector-master/Simple-Sign-Language-Detector-master/mydata/training_set',
    target_size=(227, 227),  # AlexNet input size
    batch_size=32,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    # Update with your test data directory
    'C:/Users/G NITHIN/Documents/sft_project_main/Simple-Sign-Language-Detector-master/Simple-Sign-Language-Detector-master/mydata/test_set',
    target_size=(227, 227),
    batch_size=32,
    class_mode='categorical'
)

# --- Model Training ---
model = alexnet_classifier.fit(
    training_set,
    steps_per_epoch=800,   # Adjust based on your dataset size
    epochs=25,
    validation_data=test_set,
    validation_steps=6500  # Adjust based on your dataset size
)

# --- Save the Model ---
alexnet_classifier.save('alexnet_model.keras')  
print("Model saved successfully!")

# --- Plotting ---
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(model.history['accuracy'], label='Training Accuracy')
plt.plot(model.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(model.history['loss'], label='Training Loss')
plt.plot(model.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.show()


Y_pred = alexnet_classifier.predict(test_set)
y_pred = Y_pred.argmax(axis=1)  # Convert predictions to class labels

# Get true labels from the generator
true_labels = test_set.classes

# --- Calculate Metrics ---
class_names = list(test_set.class_indices.keys()) # Get class names from generator
print(classification_report(true_labels, y_pred, target_names=class_names))

# Confusion Matrix (Optional, for deeper analysis)
conf_mat = confusion_matrix(true_labels, y_pred)
print("Confusion Matrix:\n", conf_mat)


report = classification_report(true_labels, y_pred, target_names=class_names, output_dict=True)

# Extract weighted average scores
accuracy = report['accuracy']
precision = report['weighted avg']['precision']
recall = report['weighted avg']['recall']
f1_score = report['weighted avg']['f1-score']

# Print the results
print(f"Overall Model Accuracy: {accuracy:.4f}")
print(f"Overall Model Precision (Weighted Average): {precision:.4f}")
print(f"Overall Model Recall (Weighted Average): {recall:.4f}")
print(f"Overall Model F1-Score (Weighted Average): {f1_score:.4f}")

# Confusion Matrix (Optional, for deeper analysis)
conf_mat = confusion_matrix(true_labels, y_pred)
print("Confusion Matrix:\n", conf_mat)