from keras.models import load_model
import numpy as np
from keras.preprocessing import image

# Load the trained model
classifier = load_model('Trained_model.h5')

# Function to preprocess a single image
def preprocess_image(image_path):
    try:
        test_image = image.load_img(image_path, target_size=(64, 64))  # Adjust this size if necessary
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image /= 255.0  # Normalize the image
        return test_image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

# Function to predict the sign of a single image
def predict_sign(image_path):
    test_image = preprocess_image(image_path)
    if test_image is not None:
        result = classifier.predict(test_image)
        labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        predicted_label = labels[np.argmax(result)]
        return predicted_label
    else:
        return None

# Get user input and predict
img_name = input('Enter Image Name: ')
# image_path = f'./predicting_data/{img_name}'
image_path = f'C:\\Users\\G NITHIN\\Documents\\sft_project_main\\Simple-Sign-Language-Detector-master\\Simple-Sign-Language-Detector-master\\mydata\\test_set\\A\\{img_name}'


predicted_sign = predict_sign(image_path)
if predicted_sign:
    print(f'\nPredicted Sign is: {predicted_sign}')
else:
    print('\nCould not predict the sign. Please check the image path and try again.')

# Alternatively, if you prefer the original approach with specific checks:
def detailed_predict_sign(image_path):
    test_image = preprocess_image(image_path)
    if test_image is not None:
        result = classifier.predict(test_image)
        print('Predicted Sign is:')
        print('')
        if result[0][0] == 1:
            print('A')
        elif result[0][1] == 1:
            print('B')
        elif result[0][2] == 1:
            print('C')
        elif result[0][3] == 1:
            print('D')
        elif result[0][4] == 1:
            print('E')
        elif result[0][5] == 1:
            print('F')
        elif result[0][6] == 1:
            print('G')
        elif result[0][7] == 1:
            print('H')
        elif result[0][8] == 1:
            print('I')
        elif result[0][9] == 1:
            print('J')
        elif result[0][10] == 1:
            print('K')
        elif result[0][11] == 1:
            print('L')
        elif result[0][12] == 1:
            print('M')
        elif result[0][13] == 1:
            print('N')
        elif result[0][14] == 1:
            print('O')
        elif result[0][15] == 1:
            print('P')
        elif result[0][16] == 1:
            print('Q')
        elif result[0][17] == 1:
            print('R')
        elif result[0][18] == 1:
            print('S')
        elif result[0][19] == 1:
            print('T')
        elif result[0][20] == 1:
            print('U')
        elif result[0][21] == 1:
            print('V')
        elif result[0][22] == 1:
            print('W')
        elif result[0][23] == 1:
            print('X')
        elif result[0][24] == 1:
            print('Y')
        elif result[0][25] == 1:
            print('Z')
    else:
        print('\nCould not predict the sign. Please check the image path and try again.')

# To use the detailed prediction function instead:
# detailed_predict_sign(image_path)
