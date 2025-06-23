
import numpy as np
import cv2

CLASSES = ["BACKGROUND", "AEROPLANE", "BICYCLE", "BIRD", "BOAT",
           "BOTTLE", "BUS", "CAR", "CAT", "CHAIR", "COW", "DININGTABLE",
           "DOG", "HORSE", "MOTORBIKE", "PERSON", "POTTEDPLANT", "SHEEP",
           "SOFA", "TRAIN", "TVMONITOR"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

net = cv2.dnn.readNetFromCaffe("./MobileNetSSD/MobileNetSSD.prototxt",
                               "./MobileNetSSD/MobileNetSSD.caffemodel")

cap = cv2.VideoCapture("gpOLaMo4.mp4")

if not cap.isOpened():
    print("Unable to turn on camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Unable to read frame")
        break

    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, scalefactor=0.007843, size=(300, 300), mean=127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            class_index = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = "{} [{:.2f}%]".format(CLASSES[class_index], confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[class_index], 2)
            cv2.rectangle(frame, (startX, startY - 25), (endX, startY), COLORS[class_index], cv2.FILLED)
            y = startY - 10 if startY - 10 > 10 else startY + 20
            cv2.putText(frame, label, (startX + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
