# Which-star-do-I-look-like
An openCV project which can recognize certain celebrities or can tell to whom does a face resembles closely. It is built on openCV's inbuilt face recognizer

# Make sure that you have openCV, numpy and opencv-contrib-python libraries installed.

# steps to run the program by webcam
1. Clone this repo
2. Run faceRecognitionWebCam.py
3. Press 'd' to terminate the program.

# steps to run the program by image source
1. Clone this repo
2. Open faceRecognitionFileSrc.py in editor
3. Add your source in line 17 in the file.
4. Run faceRecognitionFileSrc.py

# steps to train the program for recognizing other people
1. clone this repo
2. Add images of the person in photos/train/personName
3. Add that person's name in people list in file faceRecognitionWebCam.py, faceRecognitionFileSrc.py and faceTrain.py.
4. Delete face_trained.yml.
4. Run faceTrain.py
5. Run faceRecognitionWebCam.py or faceRecognitionFileSrc.py.
