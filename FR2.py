from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
# construct the argument parser and parse the arguments
import face_recognition
import os
import cv2
import pickle

print("[INFO] loading encodings + face detector...")
# data = pickle.loads("encodings.pickle")
infile = open("encodings.pickle",'rb')
data = pickle.load(infile, encoding='bytes')


KNOWN_FACES_DIR = 'known_faces'
UNKNOWN_FACES_DIR = 'unknown_faces'
TOLERANCE = 0.4
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'hog'  # default: 'hog', other one can be 'cnn' - CUDA accelerated (if available) deep-learning pretrained model

print("loading know faces ")

known_faces = data["encodings"]
known_names = data["names"]

def abc(img):

    # Load image
    small_frame = cv2.resize(img, (0, 0), fx=0.7, fy=0.7)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    image=rgb_small_frame

    # This time we first grab face locations - we'll need them to draw boxes
    locations = face_recognition.face_locations(image, model=MODEL)

    # Now since we know loctions, we can pass them to face_encodings as second argument
    # Without that it will search for faces once again slowing down whole process
    encodings = face_recognition.face_encodings(image, locations)

    # We passed our image through face_locations and face_encodings, so we can modify it
    # First we need to convert it from RGB to BGR as we are going to work with cv2
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match = None
        if True in results:  # If at least one is true, get a name of first of found labels
            match = known_names[results.index(True)]
            print(f"Match found : { match}")
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])

            # Get color by name using our fancy function
            color = [0,255,0]

            # Paint frame
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)

            # Paint frame
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

            # Wite a name
            cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), FONT_THICKNESS)

    cv2.imshow("filename", image)






vid = cv2.VideoCapture(0)

while (True):

    # Capture the video frame
    # by frameq
    ret, frame = vid.read()
    abc(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
