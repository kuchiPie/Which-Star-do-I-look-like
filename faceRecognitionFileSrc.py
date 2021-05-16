import numpy as np
import cv2 as cv
# Isn't that accurate XD

haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = ['Chris Evans', 'Bhumi Pednekar', 'Ranbir Kapoor', 'Sara Ali Khan', 'Elon Musk', 'Ayushmann Khurana', 'Emma Watson', 'John Abraham', 'Shraddha Kapoor', 'Varun Dhawan']


# labels = np.load('features.npy')
# features = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

# -------------Add your File source in below line----------
img = cv.imread(r'C:\Users\shrey\OneDrive\Documents\Shreyas coding\Which Star do I look like\photos\test\bhumi.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Person', gray)

# Detect the face in the image
faces_rect = haar_cascade.detectMultiScale(gray, 1.3, 6)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y : y + h, x : x + w]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'label = {people[label]} with a confidence of {confidence} % ')

    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2)
    cv.putText(img, str(people[label]), (x,y), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)


cv.imshow('detected Face', img)

cv.waitKey(0)
