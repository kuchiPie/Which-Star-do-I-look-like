import os
import numpy as np
import cv2 as cv

people = ['Chris Evans', 'Bhumi Pednekar', 'Ranbir Kapoor', 'Sara Ali Khan', 'Elon Musk', 'Ayushmann Khurana', 'Emma Watson', 'John Abraham', 'Shraddha Kapoor', 'Varun Dhawan']
DIR = r'C:\Users\shrey\OneDrive\Documents\Shreyas coding\Which Star do I look like\photos\Train'

haar_cascade = cv.CascadeClassifier('haar_face.xml')

features = []
labels = []

def create_train():
    for person in people:
        path =os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=6)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print('-------------Training Done ------------')
# print(f'Length of the features list = {len(features)}')
# print(f'Length of the labels list = {len(labels)}')

features = np.array(features, dtype = 'object')
labels = np.array(labels)


face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the Recognizer on the features the list and the labels
# list.

face_recognizer.train(features,labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)