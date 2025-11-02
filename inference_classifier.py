#inference_classifier.py
import pickle
import cv2
import mediapipe as mp
import numpy as np

# load model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0) #can change index acc to number of webcams, default 0 for mac, windows = 2

mp_hands = mp.solutions.hands
mp.drawing = mp.solutions.drawing_utils
mp.drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B'}
while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

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
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                # create an array of landmarks with the x and y of the landmarks and train the mode
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)

        x1 = int(min(x_) * W)
        y1 = int(min(y_) * H)

        x2 = int(max(x_) * W)
        y2 = int(max(y_) * H)



        #model.predict([np.asarray(data_aux)])
        # Build feature vector and match the model's expected size
        feat = np.asarray(data_aux, dtype=float).ravel()
        expected = getattr(model, "n_features_in_", len(feat))  # sklearn stores expected input size

        # pad with zeros if too short, or trim if too long
        if feat.size < expected:
            feat = np.pad(feat, (0, expected - feat.size), mode='constant')
        elif feat.size > expected:
            feat = feat[:expected]

        prediction = model.predict([feat])[0]
        # (optional) print(pred) or draw it on the frame

        predicted_character = labels_dict[int(prediction[0])]


        print(predicted_character)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.waitKey(25) #wait 25 ms between each frame

cap.release()

cv2.destroyAllWindows()







