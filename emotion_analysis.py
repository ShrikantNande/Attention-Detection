# -*- coding: utf-8 -*-

import numpy as np
import cv2
from keras.preprocessing import image
from keras.models import model_from_json
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D

style.use('fivethirtyeight')
#########################################
## Loading the Haar-Cascade Classifier from the opencv library##
#########################################
# Variable = face_cascade 
# arguments = path to haar cascade model
# Patch 1
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") #this will be in cv2 directory

###########################################
## Loadint the webCam or loading the video from the local storage##
###########################################

cap = cv2.VideoCapture(0) 
#################################################
### Loading the pre-configured model and pre-computed weights  for the  ###
### detection of faces and detection of emotions from the given video ######
model = model_from_json(open("/home/shri/Documents/emotion_video_analysis/model/facial_expression_model_structure.json", "r").read())
model.load_weights('/home/shri/Documents/emotion_video_analysis/model/facial_expression_model_weights.h5')


####################################################
##### Define the emotions to be detected ########################
### as the problem is to detect the attentive or non-attentive then we have #### 
###    classified the original 7 emotions into attentive or non-attentive #######
# variable = emotions , emotion_counts
# patch 4
emotions = ('Angry-NonAttentive', 'Disgust-NonAttentive', 'Fear-NonAttentive', 'Happy-Attentive', 'Sad-NonAttentive', 'Surprise-Attentive', 'Neutral-Attentive')
emotion_counts = {}

####################################################
### while True is that the video is loaded ##########################

###  read each frame
# patch 5 
while True:
    # read each frame 
    ret, frame = cap.read()
    # conver the frame to gray 
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # detect the faces from the grayscale frame image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
  # put the bounding box 
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        detected_face = frame[int(y):int(y+h), int(x):int(x+w)] #crop detected face
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
        detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
		
        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
		
        img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
		
        predictions = model.predict(img_pixels) #store probabilities of 7 expressions
		
		  #find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
        max_index = np.argmax(predictions[0])
		
        emotion = emotions[max_index]
        if emotion not in emotion_counts:
            emotion_counts[emotion] = 1
        else:
            emotion_counts[emotion] += 1
        
        cv2.putText(frame, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
    
    cv2.imshow('frame',frame)
    
    #cv2.imshow('grayF',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
         break

cap.release()
cv2.destroyAllWindows()

#print summary of emotions detected 
s = sum(emotion_counts.values())

percent = []
for k, v in emotion_counts.items():
    pct = v * 100.0 / s
    percent.append(pct)
    print(k, pct)

print(percent)
per = pd.Series(percent)

cols = ['emotion','value']
emo = pd.DataFrame(emotion_counts.items(),columns = cols)

emo['percent'] = per.values
import matplotlib.pyplot as plt
ax = emo.plot.bar(x='emotion', y='percent', rot=0,fontsize=5, legend=False)
plt.show()

