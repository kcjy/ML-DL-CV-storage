
import cv2
import numpy as np
import sys

sys.path.append('C:\\Users\\kenne\\Desktop\\OpenCV')

face_cascade = cv2.CascadeClassifier('C:\\Users\\kenne\\Desktop\\OpenCV\\haarcascades\\haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, 
                                          minNeighbors = 5,
                                          minSize=(30, 30))
    for (x,y,w,h) in faces:
         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
         
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#####################################################
import cv2
import numpy as np
import sys


sys.path.append('C:\\Users\\kenne\\Desktop\\OpenCV')


#cap = cv2.VideoCapture('C:\\Users\\kenne\\Desktop\\OpenCV\\Laser beam.mp4')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    #lower_red = np.array([0,50,50])
    #upper_red = np.array([10,255,255])
    #mask = cv2.inRange(hsv, lower_red, upper_red)
    
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    mask0 = cv2.inRange(hsv, lower_red, upper_red)
    
    # upper mask (170-180)
    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    
    # join my masks
    mask = mask0+mask1
    
    res = cv2.bitwise_and(frame, frame, mask=mask)
    
    
    points = cv2.findNonZero(mask)
    avg = np.mean(points, axis=0)
    
    resImage = [640, 480]
    resScreen = [1920, 1080]
    
    # points are in x,y coordinates
    pointInScreen = ((resScreen[0] / resImage[0]) * avg[0,0], (resScreen[1] / resImage[1]) * avg[0,1] )
    
    print(avg)
    
    res = cv2.bitwise_and(frame,frame, mask= mask)

    cv2.imshow('frame', frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()




