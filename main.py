#
# import cv2 as cv
# import numpy as np
# import time
# import mediapipe as mp
#
# cap = cv.VideoCapture(0)
# mpHands = mp.solutions.hands
#
# mp_drawing = mp.solutions.drawing_utils
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands()
#
# while True:
#     success, img = cap.read()
#     imgx = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#     img = cv.rectangle(img, (0, 500), (0, 500), (255, 0, 255), 10)
#
#     # To improve performance, optionally mark the image as not writeable to
#     # pass by reference.
#     results = hands.process(img)
#
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 img,
#                 hand_landmarks, mp_hands.HAND_CONNECTIONS)
#     cv.imshow('MediaPipe Hands', img)
#     cv.waitKey(0)

import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

finger_tips = [8, 12, 16, 20]
thumb_tip = 4

# like_img = cv2.imread("images/like.png")
# like_img = cv2.resize(like_img, (200, 180))
#
# dislike_img = cv2.imread("images/dislike.png")
# dislike_img = cv2.resize(dislike_img, (200, 180))

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, c = img.shape
    results = hands.process(img)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmark.landmark):
                lm_list.append(lm)
            finger_fold_status = []
            for tip in finger_tips:
                x, y = int(lm_list[tip].x * w), int(lm_list[tip].y * h)
                # print(id, ":", x, y)
                # cv2.circle(img, (x, y), 15, (255, 0, 0), cv2.FILLED)

                if lm_list[tip].x < lm_list[tip - 2].x:
                    # cv2.circle(img, (x, y), 15, (0, 255, 0), cv2.FILLED)
                    finger_fold_status.append(True)
                else:
                    finger_fold_status.append(False)

            print(finger_fold_status)

            x, y = int(lm_list[8].x * w), int(lm_list[8].y * h)
            print(x, y)

            # stop
            if lm_list[4].y < lm_list[2].y and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                    lm_list[16].y < lm_list[14].y and lm_list[20].y < lm_list[18].y and lm_list[17].x < lm_list[0].x < \
                    lm_list[5].x:
                cv2.putText(img, "STOP", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                print("STOP")

            # Forward
            if lm_list[3].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y > lm_list[10].y and \
                    lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y:
                cv2.putText(img, "FORWARD", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                print("FORWARD")

            # Backward
            if lm_list[3].x > lm_list[4].x and lm_list[3].y < lm_list[4].y and lm_list[8].y > lm_list[6].y and lm_list[
                12].y < lm_list[10].y and \
                    lm_list[16].y < lm_list[14].y and lm_list[20].y < lm_list[18].y:
                cv2.putText(img, "BACKWARD", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                print("BACKWARD")

            # Left
            if lm_list[4].y < lm_list[2].y and lm_list[8].x < lm_list[6].x and lm_list[12].x > lm_list[10].x and \
                    lm_list[16].x > lm_list[14].x and lm_list[20].x > lm_list[18].x and lm_list[5].x < lm_list[0].x:
                cv2.putText(img, "LEFT", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                print("LEFT")

            # Right
            if lm_list[4].y < lm_list[2].y and lm_list[8].x > lm_list[6].x and lm_list[12].x < lm_list[10].x and \
                    lm_list[16].x < lm_list[14].x and lm_list[20].x < lm_list[18].x:
                cv2.putText(img, "RIGHT", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                print("RIGHT")

            if all(finger_fold_status):
                # like
                if lm_list[thumb_tip].y < lm_list[thumb_tip - 1].y < lm_list[thumb_tip - 2].y and lm_list[0].x < \
                        lm_list[3].y:
                    print("LIKE")
                    cv2.putText(img, "LIKE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    # h, w, c = like_img.shape
                    # img[35:h + 35, 30:w + 30] = like_img
                # Dislike
                if lm_list[thumb_tip].y > lm_list[thumb_tip - 1].y > lm_list[thumb_tip - 2].y and lm_list[0].x < \
                        lm_list[3].y:
                    cv2.putText(img, "DISLIKE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    print("DISLIKE")
                    # h, w, c = dislike_img.shape
                    # img[35:h + 35, 30:w + 30] = dislike_img

                # rock
                if lm_list[thumb_tip].y > lm_list[thumb_tip - 1].y and lm_list[thumb_tip - 1].y > lm_list[
                    thumb_tip - 2].y and lm_list[thumb_tip].x < lm_list[3].x and lm_list[thumb_tip].y < lm_list[2].y:
                    cv2.putText(img, "rock", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    print("rock")

            mp_draw.draw_landmarks(img, hand_landmark,
                                   mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec((0, 0, 255), 6, 3),
                                   mp_draw.DrawingSpec((0, 255, 0), 4, 2)
                                   )

    cv2.imshow("Hand Sign Detection", img)
    cv2.waitKey(1)


# # # # blank = np.zeros((500, 500, 3), dtype='uint8')
# # # # # cv.imshow("zer0", blank)
# # # # cv.rectangle(blank, (0, 0), (250, 250), (0, 250, 0), thickness=2)
# # # # cv.imshow("Rectangle", blank)
# # # # cv.waitKey(3000)
# # # #
# # # # any = cv.imread('dk.jpeg')
# # # #
# # # # y = cv.cvtColor(any, cv.COLOR_BGR2GRAY)
# # # # x = cv.GaussianBlur(y, (7, 7), 0)
# # # # cv.imshow("output1", y)
# # # #
# # # # z = cv.Canny(any, 100, 100)
# # # # cv.imshow("output", z)
# # # # # cv.imshow("output2",x)
# # # # cv.waitKey(50000)
# #
# # # Import Libraries
# import cv2
# import time
# import mediapipe as mp
#
# # Grabbing the Holistic Model from Mediapipe and
# # Initializing the Model
# mp_holistic = mp.solutions.holistic
# holistic_model = mp_holistic.Holistic(
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )
#
# # Initializing the drawing utils for drawing the facial landmarks on image
# mp_drawing = mp.solutions.drawing_utils
# # (0) in VideoCapture is used to connect to your computer's default camera
# capture = cv2.VideoCapture(0)
#
# # Initializing current time and precious time for calculating the FPS
# previousTime = 0
# currentTime = 0
#
# while capture.isOpened():
#     # capture frame by frame
#     ret, frame = capture.read()
#
#     # resizing the frame for better view
#     frame = cv2.resize(frame, (800, 600))
#
#     # Converting the from BGR to RGB
#     image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     # Making predictions using holistic model
#     # To improve performance, optionally mark the image as not writeable to
#     # pass by reference.
#     image.flags.writeable = False
#     results = holistic_model.process(image)
#     image.flags.writeable = True
#
#     # Converting back the RGB image to BGR
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#
#     # Drawing the Facial Landmarks
#     mp_drawing.draw_landmarks(
#         image,
#         results.face_landmarks,
#         mp_holistic.FACEMESH_CONTOURS,
#         mp_drawing.DrawingSpec(
#             color=(255, 0, 255),
#             thickness=1,
#             circle_radius=1
#         ),
#         mp_drawing.DrawingSpec(
#             color=(0, 255, 255),
#             thickness=1,
#             circle_radius=1
#         )
#     )
#
#     # Drawing Right hand Land Marks
#     mp_drawing.draw_landmarks(
#         image,
#         results.right_hand_landmarks,
#         mp_holistic.HAND_CONNECTIONS
#     )
#
#     # Drawing Left hand Land Marks
#     mp_drawing.draw_landmarks(
#         image,
#         results.left_hand_landmarks,
#         mp_holistic.HAND_CONNECTIONS
#     )
#
#     # Calculating the FPS
#     currentTime = time.time()
#     fps = 1 / (currentTime - previousTime)
#     previousTime = currentTime
#
#     # Displaying FPS on the image
#     cv2.putText(image, str(int(fps)) + " FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
#
#     # Display the resulting image
#     cv2.imshow("Facial and Hand Landmarks", image)
#
#     # Enter key 'q' to break the loop
#     if cv2.waitKey(5) & 0xFF == ord('q'):
#         break
# #
# # When all the process is done
# # Release the capture and destroy all windows
# capture.release()
# cv2.destroyAllWindows()
# # your code goes here
# import nltk
# # import nltk
# # nltk.download('vader_lexicon')
# #
# # from nltk.sentiment import SentimentIntensityAnalyzer
# # import matplotlib.pyplot as plt
# # from matplotlib import style
# #
# # # Load the text data
# # text_data = " raone movie is good"
# #
# # # Initialize the SentimentIntensityAnalyzer object
# # analyzer = SentimentIntensityAnalyzer()
# #
# # # Analyze the sentiment of the text data
# # scores = analyzer.polarity_scores(text_data)
# #
# # # Interpret the sentiment scores and print the result
# # if scores['compound'] >= 0.05:
# #     print("The text data is positive.")
# # elif scores['compound'] <= -0.05:
# #     print("The text data is negative.")
# # else:
# #     print("The text data is neutral.")
# #
# # # Tokenize the text data
# # tokens = nltk.word_tokenize(text_data)
# #
# # # Perform part-of-speech tagging on the tokens
# # pos_tags = nltk.pos_tag(tokens)
# #
# # # Count the frequency of each part-of-speech tag
# # pos_freq = nltk.FreqDist(tag for (word, tag) in pos_tags)
# #
# # # Plot a bar chart of the part-of-speech tag frequencies
# # style.use('ggplot')
# # pos_freq.plot(title="Part-of-Speech Tag Frequencies")
# # plt.show()
#
# # import cv2 as cv
# # # import  numpy
# # #
# # img = cv.imread('k.jpeg')
# # #
# # # img = cv.rectangle(img, (0, 50), (0,0), (255, 0, 255), 10)
# # img = cv.rectangle(img, (0, 50), (0,0), (255, 0, 255), 10)
# # cv.imshow('img', img)
# # cv.waitKey(0)
# --------------------------------------------------------------------------------------
# cv.waitKey(0)import  cv2 as cv
# from  datetime import datetime
#
# current_motion=[""]
# static_back=None
# Mo_list=[]
#
# video=cv.VideroCapture(0)
#
#
# while True:
#     check,frame=video.read()
#     motion=0
#     gray=cv.cvtColor(frame,cv.COLOR_BGR)
#     gray=cv.GaussianBlur(gray,(21,21),0)
#
#     if static_back is None:
#         static_back=gray
#         continue
#
#     diff_frame=cv.absdiff(static_back,gray)
#
#     thrash_frame=cv.thershold(diff_frame,30,255,cv.THRESH_BINARY)[1]
#     thrash_frame=cv.dilate(thrash_frame.copy())
#
#     key=cv.waitKey(1)
#
#     if key==ord('q'):
#         break
#
# cv.imshow('any',any)
# import the required modules
# import cv2
# import matplotlib.pyplot as plt
# from deepface import DeepFace
#
# # read image
# img = cv2.imread('dk.jpeg')
#
# # call imshow() using plt object
# plt.imshow(img[:,:,::-1])
#
# # display that image
# plt.show()
#
# # storing the result
# result = DeepFace.analyze(img,actions=['emotion'])
#
#
#
# print(result)

# import cv2
# from deepface import DeepFace
#
# # Open webcam
# cap = cv2.VideoCapture(0)
#
# # Loop through frames
# while True:
#     # Read frame from webcam
#     ret, frame = cap.read()
#
#     # Analyze emotions using DeepFace
#     results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
#
#     result = results[0]
#     emotion = result['dominant_emotion']
#
#     # Add text to image
#     coordinates = (100, 100)
#     font = cv2.FONT_HERSHEY_SIMPLEX
# #     fontScale = 1
# #     color = (255, 0, 255)
# #     thickness = 2
# #     frame = cv2.putText(frame, emotion, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)
# #
# #     # Display frame
# #     cv2.imshow("Real-time Emotion Detection", frame)
# #
# #     # Exit on ESC
# #     if cv2.waitKey(1) == 27:
# #         break
# #
# # # Release webcam and close window
# # cap.release()
# # cv2.destroyAllWindows()
# ###############################################
# import time
#
# import time
# import cv2
# from cvzone.ClassificationModule import Classifier
#
# classifier = Classifier("xvx/keras_model.h5", "xvx/labels.txt")
#
# cap = cv2.VideoCapture(0)
#
# while True:
#     success, img = cap.read()
#     if not success:
#         print("Failed to read from video source. Exiting...")
#         break
#
#     # Pass the whole image to the classifier
#     prediction, index = classifier.getPrediction(img)
#
#     # Put the predicted class label on the original image
#     cv2.putText(img, str(prediction), (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#
#     cv2.imshow('img', img)
#     if cv2.waitKey(1) == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
# import cv2 as cv
# import urllib.request
#
# # URL to the XML file
# url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
#
# # Download the XML file and save it to a local directory
# urllib.request.urlretrieve(url, "haarcascade_frontalface_default.xml")
#
# # Open the camera device
# cap = cv.VideoCapture(0)
#
# # Loop over the frames from the camera stream
# while True:
#     # Read a frame from the camera
#     ret, frame = cap.read()
#
#     # Convert the frame to grayscale and equalize the histogram
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     gray = cv.equalizeHist(gray)
#
#     # Detect faces in the frame using the face cascade classifier
#     faces = face_cascade.detectMultiScale(gray)
#
#     # Loop over the detected faces
#     for (x, y, w, h) in faces:
#         # Draw an ellipse around the face
#         center = (x + w // 2, y + h // 2)
#         cv.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 4)
#
#         # Detect eyes in the face region using the eyes cascade classifier
#         face_gray = gray[y:y + h, x:x + w]
#         eyes = eyes_cascade.detectMultiScale(face_gray)
#
#         # Loop over the detected eyes
#         for (ex, ey, ew, eh) in eyes:
#             # Draw a circle around each eye
#             eye_center = (x + ex + ew // 2, y + ey + eh // 2)
#             radius = int(round((ew + eh) * 0.25))
#             cv.circle(frame, eye_center, radius, (255, 0, 0), 4)
#
#     # Show the resulting frame with face and eyes detection
#     cv.imshow('Capture - Face detection', frame)
#
#     # Exit the loop if the 'Esc' key is pressed
#     if cv.waitKey(1) == 27:
#         break
#
# # Release the camera and close all windows
# cap.release()
# import cv2 as cv
# cap = cv.VideoCapture(0)
#
# face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
# eye_cascade=cv.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
#
# while True:
#     ret, frame = cap.read()
#     gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
#
#     faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=4)
#
#     for (x, y, w, h) in faces:
#         cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 3)
#         roi_color=frame[y:y+h,x:x+w]
#         roi_gray=gray[y:y+h,x:x+w]
#         eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=3)
#         for(ex,ey,ew,eh)in eyes:
#             cv.rectangle(roi_color, (ex, ey), (ex +ew, ey + eh), (255, 0, 255), 2)
#
#     cv.imshow('Faces Detected', frame)
#
#     if(cv.waitKey(1)==ord('q')):
#         break

# import cv2 as cv
#
# image = cv.imread('c.jpeg')
#
# face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
# eye_cascade = cv.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
#
# gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#
# faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
#
# for (x, y, w, h) in faces:
#     cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 3)
#     roi_gray = gray[y:y+h, x:x+w]
#     roi_color = image[y:y+h, x:x+w]
#     eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=3)
#     for (ex, ey, ew, eh) in eyes:
#         cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
#
# cv.imshow('Faces Detected', image)
# cv.waitKey(0)
# cv.destroyAllWindows()


#
#
#
#
# import cv2 as cv
# cap = cv.VideoCapture(0)
#
# face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
# eye_cascade=cv.CascadeClassifier('haarcascade_eye.xml')
#
# while True:
#     ret, frame = cap.read()
#     gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
#
#     faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=4)
#
#     for (x, y, w, h) in faces:
#         cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 3)
#         roi_color=frame[y:y+h,x:x+w]
#         roi_gray=gray[y:y+h,x:x+w]
#         eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=3)
#         for(ex,ey,ew,eh)in eyes:
#             cv.rectangle(roi_color, (ex, ey), (ex +ew, ey + eh), (255, 0, 0), 2)
#
#     cv.imshow('Faces Detected', frame)
#
#     if(cv.waitKey(1)==ord('q')):
#         break
# -----------hhaaaaaaaaaaaaaaaaaa-------------------------------------------------------
#
# import cv2 as cv
#
# image = cv.imread('m.png')
#
# face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
# eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
#
# gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#
# faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
#
# for (x, y, w, h) in faces:
#     cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 3)
#     roi_gray = gray[y:y+h, x:x+w]
#     roi_color = image[y:y+h, x:x+w]
#     eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=3)
#     for (ex, ey, ew, eh) in eyes:
#         cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 2)
#
# cv.imshow('Faces Detected', image)
# cv.waitKey(0)
# cv.destroyAllWindows()
#
#
#
#
# # Draw rectangles around the detected faces
#
#
# # Display the output image
#
#
#
#
#
#
#
#
# import cv2
# import mediapipe as mp
#
# # Initialize MediaPipe Hand model and drawing utilities
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
#
# # Initialize video capture
# cap = cv2.VideoCapture(0)
#
# # Set up MediaPipe Hand model
# with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
#     # Loop over video frames
#     while cap.isOpened():
#
#         # Read a frame from the video stream
#         success, image = cap.read()
#         if not success:
#             break
#
#         # Convert image to RGB format and detect hands
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         results = hands.process(image)
#
#         # Draw hand landmarks and connect them with lines
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(
#                     image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#
#         # Display the resulting image
#         cv2.imshow('Hand Gestures', image)
#         if cv2.waitKey(5) & 0xFF == 27:
#             break
#
# # Release video capture and close all windows
# cap.release()
# cv2.destroyAllWindows()
#####################################################
# import cv2
# import mediapipe as mp
#
# # Initialize MediaPipe Hands model and drawing utility
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
#
# # Initialize video capture using default camera (index 0)
# cap = cv2.VideoCapture(0)
#
# # Set minimum detection and tracking confidences for the model
# min_detection_confidence = 0.5
# min_tracking_confidence = 0.5
#
# # Initialize MediaPipe Hands model as a context manager
# with mp_hands.Hands(
#     min_detection_confidence=min_detection_confidence,
#     min_tracking_confidence=min_tracking_confidence
# ) as hands:
#
#     # Loop over video frames until the user presses the 'Esc' key
#     while True:
#
#         ret, frame = cap.read()
#
#         if not ret:
#             break
#
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = hands.process(image)
#         H,W,z=image.shape
#
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 if hand_landmarks is not None:
#                     for id, landmark in enumerate(hand_landmarks.landmark):
#                         x_coord = int(landmark.x * W)
#                         y_coord = int(landmark.y * H)
#
#                         print(f'Hand {id} landmark {landmark}: ({x_coord}, {y_coord}')
#
#                         if id%2==0:
#
#                           cv2.circle(image, (x_coord, y_coord), 12, (255, 0, 0), cv2.FILLED)
#
#                         else:
#
#                           cv2.circle(image, (x_coord, y_coord), 12, (0, 0, 153), cv2.FILLED)
#
#
#
#                     mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#
#         # Show the resulting image in a window named 'Hand Gestures'
#         cv2.imshow('Hand Gestures', image)
#
#         # Exit the loop if the user presses the 'Esc' key
#         if cv2.waitKey(1) == 27:
#             break
#
# # Release the video capture and close all windows
# cap.release()
# cv2.destroyAllWindows()
# STEP 1: Import the necessary modules.
# import cv2
# import mediapipe as mp
#
# # STEP 2: Create a HandLandmarker object.
# mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2)
# # STEP 3: Open the video stream.
# cap = cv2.VideoCapture(0)
#
# # STEP 4: Process each frame of the video stream.
# while True:
#     # Read a frame from the video stream.
#     ret, frame = cap.read()
#
#     # Detect hand landmarks from the frame.
#     results = mp_hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#
#     # Detect thumbs up gesture from the hand landmarks.
#     # Detect thumbs up and thumbs down gestures from the hand landmarks.
#     thumbs_up = False
#     thumbs_down = False
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             # Get the landmark points for the thumb and index finger.
#             thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
#             thumb_tip1 = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
#
#             index_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
#
#             # Check if the thumb is to the left and above the index finger for thumbs up.
#             if thumb_tip.x > index_finger_tip.x and thumb_tip.y > index_finger_tip.y:
#                 thumbs_up = True
#
#             # Check if the thumb is to the right and below the index finger for thumbs down.
#             if thumb_tip.x < index_finger_tip.x and thumb_tip.y >index_finger_tip.y:
#                 thumbs_down = True
#
#             # Draw the hand landmarks on the frame.
#             for landmark_point in hand_landmarks.landmark:
#                 x = int(landmark_point.x * frame.shape[1])
#                 y = int(landmark_point.y * frame.shape[0])
#                 cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
#
#     # Draw the thumbs up and thumbs down gesture on the frame.
#     if thumbs_up:
#         cv2.putText(frame, 'Thumbs down', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#     elif thumbs_down:
#         cv2.putText(frame, 'Thumbs up', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#
#     # Display the frame with the hand landmarks and thumbs up gesture.
#     cv2.imshow('Hand Landmarks', frame)
#
#     # Press 'q' to exit the program.
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release the video capture object and close all windows.
# cap.release()
# cv2.destroyAllWindows()



