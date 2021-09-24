import cv2
import numpy as np
import face_recognition

imgElon = face_recognition.load_image_file('images/elon1.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('images/bill1.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]

cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255,255,255), 2)

results = face_recognition.compare_faces([encodeElon], encodeTest)[0]
faceDis = face_recognition.face_distance([encodeElon], encodeTest)[0]

print(encodeElon)

# print(results, faceDis)
cv2.putText(imgTest, f"{results} {round(faceDis, 2)}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))


cv2.imshow('Elon', imgElon)
cv2.imshow('Elon test', imgTest)
cv2.waitKey(0)