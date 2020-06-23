
import cv2
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, callbacks
import numpy as np
import os
import time
import nets

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

model_path = './models/model1/model.h5'
model = models.load_model(model_path)
input_size = 42

### Add if GPU needed
# # set gpu auto allocate memory
# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# for gpu in gpus:
#   tf.config.experimental.set_memory_growth(gpu,True)

### Camara
cap = cv2.VideoCapture(0)
haarcascade_path = '/home/dean/Projects/Emotion_Detect/.venv/lib64/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml'
detector = cv2.CascadeClassifier(haarcascade_path)
emotion = 'Neutral'
isRecording = False

### Main loop
while(True):
    ret,frame = cap.read()
    
    # cv2 detector
    face_zone = detector.detectMultiScale(frame)
    cv2.putText(frame,"Q to quit", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    
    for x,y,w,h in face_zone:
        cv2.putText(frame, emotion, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.rectangle(frame, (x,y), (x+w,y+h), [255,0,0], 2)
        frame_slice = frame[y:y+h, x:x+w]
        frame_slice = cv2.resize(frame_slice, (48,48))
        # cv2.imshow('frame_slice',frame_slice) # enable to see the slice
        frame_slice = cv2.cvtColor(frame_slice, cv2.COLOR_BGR2GRAY) # gray
        frame_slice = frame_slice.reshape(1, 48, 48)
        tempFace = np.concatenate(
            (frame_slice[:, 0:0+input_size, 0:0+input_size],
             frame_slice[:, 0:0+input_size, 6:6+input_size],
             frame_slice[:, 6:6+input_size, 0:0+input_size],
             frame_slice[:, 6:6+input_size, 6:6+input_size],
             frame_slice[:, 3:3+input_size, 3:3+input_size])
        )
        tempFace = np.concatenate((tempFace, np.flip(tempFace, 2)))
        
        # Detection
        tempFace = tempFace / 255.0
        with tf.device('/cpu:0'):
            preds = np.argmax(model.predict(np.resize(tempFace,(10,input_size,input_size,1))), axis=-1)
        counts = np.bincount(preds)
        label = np.argmax(counts)
        emotion = class_names[label]
    
    # Record
    if isRecording:
        videoWriter.write(frame) # write
        cv2.putText(frame,"Recording", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow('frame',frame)
    
    key = cv2.waitKey(1)&0xFF # Key input
    
    if key == ord('q'):
        if isRecording:
            videoWriter.release()
            cmd = 'ffmpeg -i ' \
                + video_filename \
                + ' -vcodec libx265 -b:v 1.5M ' \
                +'videos/' + time_str + '_c.mp4 ' \
                +'-loglevel quiet -y' # quiet run, overlap
            cmd_return = os.system(cmd)
            if cmd_return == 0:
                cmd = 'rm ' + video_filename
                os.system(cmd)
            else:
                print('Failed to compress the video')
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
            cv2.putText(frame,"Saving", (300,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cmd = 'ffmpeg -i ' \
                + video_filename \
                + ' -vcodec libx265 ' \
                +'videos/' + time_str + '_c.mp4 ' \
                +'-loglevel quiet -y &' # quiet run, overlap, and running background
            cmd_return = os.system(cmd)
            if cmd_return == 0:
                cmd = 'rm ' + video_filename
                os.system(cmd)
            else:
                print('Failed to compress the video')
            
        isRecording = not isRecording

