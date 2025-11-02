import os
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

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

        plt.figure()
        plt.imshow(img_rgb)

plt.show()



