
import cv2
# import csv
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, callbacks
import numpy as np
import os
import time
import nets

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# model = nets.getnet1()
model = nets.getnet2()


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

""" checkpoint_path1 = "./models_2/model.ckpt"
checkpoint_dir1 = os.path.dirname(checkpoint_path1)
latest = tf.train.latest_checkpoint(checkpoint_dir1) """

checkpoint_path2 = "./models_3/model.ckpt"
checkpoint_dir2 = os.path.dirname(checkpoint_path2)
latest = tf.train.latest_checkpoint(checkpoint_dir2)

model.load_weights(latest)

### Camara
cap = cv2.VideoCapture(0)
iter_num = 10
tempFaces = np.zeros((iter_num,48,48))
count = 0
haarcascade_path = '/home/dean/Projects/Emotion_Detect/.venv/lib64/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml'
detector = cv2.CascadeClassifier(haarcascade_path)
emotion = 'Neutral'

# save video
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# # fps = cap.get(cv2.CAP_PROP_FPS)
# fps = 5
# size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
isRecording = False
# date_str = time.strftime("%Y_%m_%d/%Y_%m_%d", time.localtime())
# time_str = time.strftime("%H_%M_%S", time.localtime())
# video_filename = 'videos/' + time_str + '.avi'
# video_dir = 'videos/' + date_str
# if not os.path.exists(video_dir):
#     os.makedirs(video_dir)
# videoWriter = cv2.VideoWriter(video_filename, fourcc, fps, size)

while(True):
    ret,frame = cap.read()
    
    ## cv2 detector
    face_zone = detector.detectMultiScale(frame)
    cv2.putText(frame,"Q to quit", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0))
    for x,y,w,h in face_zone:
        cv2.putText(frame, emotion, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
        cv2.rectangle(frame, pt1=(x,y), pt2=(x+w,y+h), color=[0,0,255])
        frame_slice = frame[y:y+h, x:x+w]
        frame_slice = cv2.resize(frame_slice, (48,48))
        # cv2.imshow('frame_slice',frame_slice) # enable to see the slice
        
        tempFaces[count] = cv2.cvtColor(frame_slice,cv2.COLOR_BGR2GRAY) # gray and save
        count = (count+1)%iter_num
        
        # detection
        if count == 0:
            tempFaces = tempFaces / 255.0
            with tf.device('/cpu:0'):
                # preds = np.argmax(model.predict(tempFaces), axis=-1)
                preds = np.argmax(model.predict(np.resize(tempFaces,(iter_num,48,48,1))), axis=-1)
            counts = np.bincount(preds)
            label = np.argmax(counts)
            emotion = class_names[label]
        
    cv2.imshow('frame',frame)
    
    if isRecording:
        videoWriter.write(frame) # write
    
    key = cv2.waitKey(1)&0xFF
    
    if key == ord('q'):
        if isRecording:
            videoWriter.release()
        break
    elif key == ord('r'):
        if not isRecording:
            # start Recording
            time_str = time.strftime("%Y_%m_%d/%Y_%m_%d_%H_%M_%S", time.localtime())
            video_filename = 'videos/' + time_str + '.avi'
            video_dir = os.path.dirname(video_filename)
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = cap.get(cv2.CAP_PROP_FPS)
            # fps = 10
            size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            videoWriter = cv2.VideoWriter(video_filename, fourcc, fps, size)
            
        else:
            # stop Recording
            videoWriter.release()
        isRecording = not isRecording
            
    # if cv2.waitKey(1)&0xFF==ord('q'):
    #     if isRecording:
    #         videoWriter.release()
    #     break



