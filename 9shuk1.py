from keras.models import load_model
import numpy as np
from keras.utils import plot_model
import math
import glob
import os
import cv2
model_path = 'facenet_keras.h5'
model = load_model(model_path)


#model.load_weights('D:/keras-facenet-master/model/facenet_keras_weights.h5')
cascade_path = 'haarcascade_frontalface_alt2.xml'
#model.summary()

def prewhiten(x):

    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj

    return y
def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output


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




def dist(ofg):
    print(ofg.ndim)
    ofg = calc_embs(ofg,10,1)
    for y in predict:

        dot = np.sum(np.multiply(y, ofg), axis=1)
        norm = np.linalg.norm(y, axis=1) * np.linalg.norm(ofg, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    print(dist)


def search_face( frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop through all the faces detected and determine whether or not they are in the database
    identities = []
    for (x, y, w, h) in faces:
        x1 = x-PADDING
        y1 = y-PADDING
        x2 = x+w+PADDING
        y2 = y+h+PADDING

        frame = cv2.rectangle(frame,(x1, y1),(x2, y2),(255,0,0),2)
    return frame




database = []
labels = []
imagePath = "images/"
jpg1 = ".jpg"
#predict = []
name = []
i = int()
predict = np.zeros((16,128))

for file in glob.glob("images/*"):
    identify = os.path.splitext(os.path.basename(file))[0]
    #print (identify)
    cv = imagePath + identify + jpg1
    print (cv)
    img = (cv2.imread(cv))
    img = cv2.resize(img,(160,160))
    new = calc_embs(img, 10, 1)
    #print(new)
    predict[i:] = new
    name.append(identify)
    i = i+1
    #database = np.array(img)

print(name)
#print(labels)
print(predict.shape)
    #database[identify] = recognation(file)


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
PADDING = 50
i = int()

dist1 = int()
zz = int(0)
h = int()
zz = 0
pre = np.zeros((1,128))
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

u = int(0)
i = int(0)
import telebot
import cv2
import serial
import matplotlib.pyplot as plt
ser = serial.Serial('COM3', 9600, write_timeout=1, timeout=0.1)  
print(ser.name)         # check which port was really used
##sio = io.TextIOWrapper(io.BufferedRWPair(ser, ser))
#ser.close()
cap = cv2.VideoCapture(0)
zz = 0
while(True):
# Capture frame-by-frame
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

