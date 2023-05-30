import cv2

img = cv2.imread('test1.jpg')
grey_im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
faces_rect = face_cascade.detectMultiScale(grey_im, 1.1, 9)
eye_rect = eye_cascade.detectMultiScale(grey_im, 1.1, 9)

for (x, y, w, h) in faces_rect:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

for (x, y, w, h) in eye_rect:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

cv2.imshow('Detected', img)
cv2.waitKey(0)
