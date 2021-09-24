import os
import numpy as np
import cv2
import face_recognition
from datetime import datetime


path = 'datas'
images = []
classeNames = []
myList = os.listdir(path)


for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classeNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtStr = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name}, {dtStr}')

def drawRectangle(faceLoc):
    y1, x2, y2, x1 = faceLoc
    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.rectangle(img, (x1, y2), (x2, y2 + 35), (0, 255, 0), -1)
    cv2.putText(img, name, (x1 + 6, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

encodeListKnow = findEncodings(images)

print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()
    img = cv2.flip(img, 1)
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    # faceCurFrame = cv2.cvtColor(faceCurFrame, cv2.COLOR_RGB2GRAY)

    encodesCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnow, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnow, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classeNames[matchIndex].upper()
            print(name)
            markAttendance(name)
            drawRectangle(faceLoc)
        else:
            name = "Unknow"
            drawRectangle(faceLoc)
    cv2.imshow('camera', img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break