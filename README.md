<h1>SmartBox - box with face recognition function (Keras and Arduino)</h1>
<img src="https://github.com/beetlea/face_reco_camera/blob/master/%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA.JPG?raw=true"></src>
Facial recognition has already taken over the whole world. All major countries are already using this useful feature. Why not make people's lives even more convenient and embed facial recognition in the luggage storage?

<h2>To do this, we will need</h2>

<li>downloaded neural network facenet
<li>computer
<li>keras
<li>opencv

From the beginning we import dependencies
<code>
from keras.models import load_model
import numpy as np
from keras.utils import plot_model
import math
import glob
import os
import cv2
import serial
</code>

Then we load the grid and specify the path for the face detector
<code>
model_path = 'facenet_keras.h5'
model = load_model(model_path)
cascade_path = 'haarcascade_frontalface_alt2.xml'
</code>

A function that formats an image and runs it through a neural network
<details>
    <code>
def calc_embs(imgs, margin, batch_size):
    fram1e = cv2.resize(imgs,(160,160))
    ofg2 = np.array(fram1e)
    aligned_images = prewhiten(ofg2)
    pd = []
    x_train = np.array([aligned_images])
    embs1 = model.predict_on_batch(x_train)
    embs1.reshape(1,-1)
    embs = l2_normalize(np.concatenate(embs1))
    return embs   
</code>
</details>
    
A function that, when a button is pressed for the first time, saves the face of a person who has passed through the neural network, and the second time, having driven a new face through the network, compares it with the saved one
<details>
<code>
def reco_face(frame, i):

    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = frame
    #i = 0
    h = 0
    v = 0
    u = 0
    name_out = 'я тебя незнаю'
    #print(ofg.shape)
    #img = search_face(img, frame, face_cascade)
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    print(faces)
    if faces == ():
        v = 5
    # Loop through all the faces detected and determine whether or not they are in the database
    identities = []
    for (x, y, w, h) in faces:
        x1 = x-PADDING
        y1 = y-PADDING
        x2 = x+w+PADDING
        y2 = y+h+PADDING

        frame = cv2.rectangle(frame,(x1, y1),(x2, y2),(255,0,0),2)
        height, width, channels = frame.shape
        # The padding is necessary since the OpenCV face detector creates the bounding box around the face and not the head
        part_image = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
        if i == 1:
            pre[0:] = calc_embs(part_image,10,1)
            while u!=1:

                u = ser.write( b'P') 
            u=0  
        else: ofg = calc_embs(part_image,10,1)
        #print(ofg)
        #i = i + 1
        if i > 1:
            for m in pre:
                dot = np.sum(np.multiply(m, ofg), axis=0)
                norm = np.linalg.norm(m, axis=0) * np.linalg.norm(ofg, axis=0)
                similarity = dot / norm
                dist1 = np.arccos(similarity) / math.pi
                if dist1<0.32:
                    print(dist1)
                    h = 1
    return h,v
</code>
</details>

Well, the main. It all starts with the arduino, when the letter B is fed through the uart, which means that the button is pressed. Next, a command is sent to the arduino to open the drawer and the face recognition and preservation function is launched. Then, if the command to press the button came from the arduino again, we run the recognition function again and if the faces converge, then open the box.
<details>
    <code>
ser = serial.Serial('COM3', 9600, write_timeout=1, timeout=0.1)  
print(ser.name)         # check which port was really used
cap = cv2.VideoCapture(0)
zz = 0
while(True):
    ret, frame = cap.read()
    frame1 = search_face(frame)
    cv2.imshow('ffff', frame1)
    ff=ser.read(1)
    if(ff == b'B'):
        print("press_button")
        ff = b'u'
        zz = zz + 1
        mmm, f = reco_face(frame, zz)
        if f == 5:
            zz = 0
        print(mmm)
        if mmm == 1:
            print("otkrivaio")    
            while u!=1:
                u = ser.write( b'P') 
            u=0  
            h = 0  
            zz = 0     
    if cv2.waitKey(33) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
</code>
</details>
    
<h2>Video</h2>
https://www.youtube.com/watch?v=iWQTNod7UKw&t
