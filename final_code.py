import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image
import os
from PIL import Image
import time
import pickle
from playsound import playsound
import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from gtts import gTTS
import tempfile
import base64
from io import BytesIO
# from keras.preprocessing import image
from tensorflow.keras.utils import load_img, img_to_array
# Load the pre-trained model
classifier = load_model('Trained_model.h5')

image_x, image_y = 64, 64


# def text_to_speech(text, lang='en'):
#     # Create a gTTS object
#     tts = gTTS(text=text, lang=lang, slow=False)
    
#     # Save the converted audio to a file
#     audio_file = "output.mp3"
#     tts.save(audio_file)
    
#     # Play the converted audio file
#     playsound(audio_file)

def text_to_speech(text_input, lang='en'):
    tts = gTTS(text=text_input, lang='en')

    # Store audio in memory (no temporary file)
    fp = BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)  # Reset pointer to the beginning

    # Encode as base64 string for embedding in HTML
    b64 = base64.b64encode(fp.getvalue()).decode()

    # Embed audio player directly in the app
    st.markdown(f"""
    <audio controls autoplay>
        <source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg"> 
    </audio>
    """, unsafe_allow_html=True)

def predictor(image_path='1.png'):
    test_image = load_img(image_path, target_size=(64, 64))
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)
    
    for i in range(26):
        if result[0][i] == 1:
            return chr(65 + i)
    return ''




# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


# model_dict = pickle.load(open('./model.p', 'rb'))
# model = model_dict['model']
# # Define a dictionary mapping class indices to labels
# labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'} 
# my_list = []




st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('Sign Language Detection')
#st.sidebar.subheader('-Parameter')



app_mode = st.sidebar.selectbox('Choose the App mode',
['About App','Sign Language to Text','Text to sign Language']
)

if app_mode =='About App':
    st.title('Real-Time Sign Language Recognition')
    # st.header("Bridging Communication with AI-Powered Sign Recognition")
    st.markdown('This application translates sign language into text in real-time using computer vision and machine learning ')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )
    
    

    # Project Description
    st.write("This application helps bridge communication gaps by translating sign language into text in real-time. Our goal is to promote inclusivity and accessibility.")

    # Features
    st.subheader("Features")
    st.write("* Real-time sign recognition using your camera")
    st.write("* Text display of recognized words and signs")

    # How to Use
    st.subheader("How to Use")
    st.write("1. Open the application and allow camera access.")
    st.write("2. Point your camera towards the person signing, ensuring good lighting and hand visibility.")
    st.write("3. Recognized signs will appear as text on the screen.")

    # Call to Action
    st.subheader("Help Us Improve")
    st.write("Your feedback is valuable!  Please let us know how we can make this application even better.")
    
    
elif app_mode == 'Sign Language to Text':
    print("hello")
    st.title("Sign Language Recognition")
    st.header("Translate Sign Language in Real-Time")

    run_app = st.checkbox("Run Application")
    if run_app:
        prediction_text = st.empty() 
        word_display = st.empty() 
        # predicted_character = ""  
        word=""
        last_detection_time = time.time()
        word_detction_time=time.time()
        print("hello")
        st.title("Real-Time Hand Gesture Recognition")
    
        # Create trackbars for adjusting HSV values (visible to the user)
        st.sidebar.header("Adjust HSV Range")
        l_h = st.sidebar.slider("L - H", 0, 179, 0)
        l_s = st.sidebar.slider("L - S", 0, 255, 50)
        l_v = st.sidebar.slider("L - V", 0, 255, 50)
        u_h = st.sidebar.slider("U - H", 0, 179, 179)
        u_s = st.sidebar.slider("U - S", 0, 255, 255)
        u_v = st.sidebar.slider("U - V", 0, 255, 255)
        
        lower_hsv = np.array([l_h, l_s, l_v])
        upper_hsv = np.array([u_h, u_s, u_v])
        
        # Start video capture
        cap = cv2.VideoCapture(0)
        
        stframe = st.empty()
        stframe_mask = st.empty()
        
        while True:
            
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            roi = frame[100:300, 425:625]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
            
            save_img = cv2.resize(mask, (image_x, image_y))
            cv2.imwrite("1.png", save_img)
            
            img_text = predictor()
            
            # Display mask within the Streamlit interface
            stframe_mask.image(mask, channels="GRAY", caption="Mask")
            
            # Draw rectangle and put text on the frame
            cv2.rectangle(frame, (425, 100), (625, 300), (0, 255, 0), 2)
            cv2.putText(frame, img_text, (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0), 2)

            if time.time() - word_detction_time >20:
                word_str=str(word) 
                text_to_speech(word_str)

                        # word = word.join(predicted_character) 
                word = " "   
                        #word_display.text(f"Word: {word}") 
                word_detction_time = time.time()
                        
                last_detection_time = time.time()

            if time.time() - last_detection_time > 5: 
                        # word = word.join(predicted_character) 
                word = word + img_text   
                        #word_display.text(f"Word: {word}") 
                        
                last_detection_time = time.time()
            
            stframe.image(frame, channels="BGR")
            word_display.text(f"Predicted Word: {word} ")
            
            if cv2.waitKey(1) == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        # frame_placeholder = st.empty()  
        # prediction_text = st.empty() 
        # word_display = st.empty() 
        # predicted_character = ""  
        # word=""
        # last_detection_time = time.time()
        # word_detction_time=time.time()

        # # Try opening the camera
        # cap = None
        # for i in range(10):
        #     cap = cv2.VideoCapture(i)
        #     if cap.isOpened():
        #         break

        # if not cap.isOpened():
        #     st.error("Failed to open any camera.")
            

        # # Main loop 
        # while run_app:
        #     ret, frame = cap.read()            
        #     if not ret:
        #         st.error("Error: Unable to read frame from camera")
        #         break

        #     # Get dimensions of the frame
        #     H, W, _ = frame.shape

        #     # Convert frame from BGR to RGB color space
        #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #     # Process the frame using MediaPipe Hands
        #     results = hands.process(frame_rgb)

        #     # Check if hand landmarks were detected in the frame
        #     if results.multi_hand_landmarks:
        #         for hand_landmarks in results.multi_hand_landmarks: 
        #             # Draw hand landmarks on the frame
        #             mp_drawing.draw_landmarks(
        #                 frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
        #                 mp_drawing_styles.get_default_hand_landmarks_style(),
        #                 mp_drawing_styles.get_default_hand_connections_style())

        #             # Extract hand landmark coordinates & calculate data_aux
        #             data_aux = [] 
        #             x_ = []
        #             y_ = []
        #             for i in range(len(hand_landmarks.landmark)):
        #                 x = hand_landmarks.landmark[i].x
        #                 y = hand_landmarks.landmark[i].y
        #                 x_.append(x)
        #                 y_.append(y)
        #                 data_aux.append(x - min(x_))
        #                 data_aux.append(y - min(y_))

        #             # Make a prediction 
        #             prediction = model.predict([np.asarray(data_aux)])
        #             predicted_character = labels_dict[int(prediction[0])] 
        #             if time.time() - word_detction_time >40: 
        #                 # word = word.join(predicted_character) 
        #                 word = " "   
        #                 #word_display.text(f"Word: {word}") 
        #                 word_detction_time = time.time()
                        
        #                 last_detection_time = time.time()
        #             if time.time() - last_detection_time > 6: 
        #                 # word = word.join(predicted_character) 
        #                 word = word + predicted_character   
        #                 #word_display.text(f"Word: {word}") 
                        
        #                 last_detection_time = time.time()

        #             # Draw bounding box & display prediction
        #             x1 = int(min(x_) * W) - 10
        #             y1 = int(min(y_) * H) - 10
        #             x2 = int(max(x_) * W) - 10
        #             y2 = int(max(y_) * H) - 10
        #             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        #             cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
        #                         cv2.LINE_AA)

        #     # Update the Streamlit interface 
        #     frame_placeholder.image(frame)  
        #     prediction_text.text(f"Predicted Sign: {predicted_character}")
        #     word_display.text(f"Predicted Word: {word} ")
        # cap.release()
    
else:
    st.markdown('Text to Sign Language (The System use Indian Sign Language)')


    # define function to display sign language images
    def display_images(text):
        # get the file path of the images directory
        img_dir = "images/"

        # initialize variable to track image position
        image_pos = st.empty()

        # iterate through the text and display sign language images
        for char in text:
            if char.isalpha():
                # display sign language image for the alphabet
                img_path = os.path.join(img_dir, f"{char}.png")
                img = Image.open(img_path)

                # update the position of the image
                image_pos.image(img, width=500)

                # wait for 2 seconds before displaying the next image
                time.sleep(1)

                # remove the image
                image_pos.empty()
            elif char == ' ':
                # display space image for space character
                img_path = os.path.join(img_dir, "space.png")
                img = Image.open(img_path)

                # update the position of the image
                image_pos.image(img, width=500)

                # wait for 2 seconds before displaying the next image
                time.sleep(1)

                # remove the image
                image_pos.empty()

        # wait for 2 seconds before removing the last image
        time.sleep(2)
        image_pos.empty()


    text = st.text_input("Enter text:")
    # convert text to lowercase
    text = text.lower()

    # display sign language images
    display_images(text)
