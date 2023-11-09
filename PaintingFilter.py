import cv2
import imutils

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
k = False  # Cambiamos k a una variable booleana

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    p = cv2.waitKey(1)

    if p == 97: 
        k = not k  

    if k:  
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Result', frame)

    if p == 27:  
        break

cap.release()
cv2.destroyAllWindows()
