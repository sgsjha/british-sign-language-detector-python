#inference_classifier.py

import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0) #can change index acc to number of webcams, default 0 for mac, windows = 2

mp_hands = mp.solutions.hands
mp.drawing = mp.solutions.drawing_utils
mp.drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

while True:
    ret, frame = cap.read()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    # iterate through results - show hands : DRAW LANDMARKS
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.drawing.draw_landmarks(
                frame,  # img to draw
                hand_landmarks,  # output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp.drawing_styles.get_default_hand_landmarks_style(),
                mp.drawing_styles.get_default_hand_connections_style()
            )


    cv2.imshow('frame', frame)
    cv2.waitKey(25) #wait 25 ms between each frame

cap.release()

cv2.destroyAllWindows()







