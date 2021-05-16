import cv2 as cv
import numpy as np

haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = ['Chris Evans', 'Bhumi Pednekar', 'Ranbir Kapoor', 'Sara Ali Khan', 'Elon Musk', 'Ayushmann Khurana', 'Emma Watson', 'John Abraham', 'Shraddha Kapoor', 'Varun Dhawan']


# labels = np.load('features.npy')
# features = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

vid = cv.VideoCapture(0)

while True:

    ret, frame = vid.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # cv.imshow('Person', gray)

    # Detect the face in the image
    faces_rect = haar_cascade.detectMultiScale(gray, 1.3, 6)
    cv.putText(frame, 'press d to terminate', (frame.shape[1]//4, frame.shape[0]//4), cv.FONT_HERSHEY_PLAIN, 1.0, (0,0,255), thickness=2) 

    for (x,y,w,h) in faces_rect:
        faces_roi = gray[y : y + h, x : x + w]

        label, confidence = face_recognizer.predict(faces_roi)
        print(f'label = {people[label]} with a confidence of {confidence} % ')

        cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), thickness=2)
        cv.putText(frame, str(people[label]), (x,y), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)




    cv.imshow('Video', frame)

    if cv.waitKey(1) & 0xFF==ord('d'):
        break

vid.release()
cv.destroyAllWindows()
