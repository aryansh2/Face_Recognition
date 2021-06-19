import os
import cv2
import face_recognition
import numpy as np
from datetime import datetime

path = 'Images'
images = []
names = []
classList = os.listdir(path)
print(classList)
for cl in classList:
    img = cv2.imread(f'{path}/{cl}')
    images.append(img)
    names.append(os.path.splitext(cl)[0])
print(names)


def encodings(images):
    encodes = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodes.append(encode)
    return encodes


encodesKnown = encodings(images)
print(len(encodesKnown))


def markAttendance(name):
  with open('venv/Attendance.csv','r+') as f:
      list=f.readlines()
      nameList=[]
      for line in list:
          entry=line.split(',')
          nameList.append(entry[0])
      if name not in nameList:
          time=datetime.now()
          dstring=time.strftime('%H:%M:%S')
          f.writelines(f'\n{name},{dstring}')

cap = cv2.VideoCapture(0)
while True:
    _, img1 = cap.read()
    img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    faceLoc = face_recognition.face_locations(img2)
    encodes1 = face_recognition.face_encodings(img2, faceLoc)
    for encodies, faceloc in zip(encodes1, faceLoc):
        matches = face_recognition.compare_faces(encodesKnown, encodies)
        dist = face_recognition.face_distance(encodesKnown, encodies)
        print(dist)
        index = np.argmin(dist)
        if names[index]:
            y1, x2, y2, x1 = faceloc
            cv2.rectangle(img1, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.rectangle(img1, (x1, y2 - 25), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img1, names[index], (x1 + 6, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            markAttendance(names[index])
    cv2.imshow('imgs', img1)
    cv2.waitKey(1)
