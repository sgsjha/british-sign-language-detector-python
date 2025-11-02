import os
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands
mp.drawing = mp.solutions.drawing_utils
mp.drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)



#iterate through the photos

#save in a file to later train the model

DATA_DIR = './data'

for dir_ in os.listdir(DATA_DIR):

    # skip files like .DS_Store
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue

    for img_path in os.listdir(os.path.join(DATA_DIR, dir_))[:1]:
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        # need to covert img to rgb to input into mediapipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        #iterate through results - show hands : DRAW LANDMARKS
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                #To show the drawing of landmarks : TESTING
                 mp.drawing.draw_landmarks(
                    img_rgb, #img to draw
                    hand_landmarks, #output
                    mp_hands.HAND_CONNECTIONS, #hand connections
                    mp.drawing_styles.get_default_hand_landmarks_style(),
                    mp.drawing_styles.get_default_hand_connections_style()
                )


        plt.figure()
        plt.imshow(img_rgb)

plt.show()



