import cv2 as cv
import numpy as np
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.FaceMeshModule import FaceMeshDetector 
from sqlalchemy import true

#NOTICE : If you have cemara opening error change 'cap' number to 0 or other numbers
cap = cv.VideoCapture(1)
detector = FaceDetector()
meshdetector = FaceMeshDetector(maxFaces=2)

while (true):
    rc, frame = cap.read()

    frame, bbox = detector.findFaces(frame)
    frame, faces = meshdetector.findFaceMesh(frame)
    #If you don't want to show logs comment line 18 and 19
    if bbox:
       print (bbox)
    
    cv.imshow('Face Detector ... (exit = Esc)', frame)
    
    keyexit =  cv.waitKey(5) & 0xFF
    if keyexit == 27:
        break

cv.destroyAllWindows()
cap.release()
