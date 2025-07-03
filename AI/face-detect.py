import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)

while True:
    success, frame = cap.read()
    results = mp_face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if results.detections:
        print(f"Faces Detected: {len(results.detections)}")
