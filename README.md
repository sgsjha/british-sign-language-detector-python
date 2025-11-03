<div align=\"center\">

# 🤟 BSL Alphabet Detector - Python

A real-time British Sign Language (BSL) alphabet detector built using Python, MediaPipe, OpenCV, and Scikit-Learn.

[![Python](https://img.shields.io/badge/Python3%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-34A853?style=for-the-badge&logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Windows%20%7C%20Linux-000000?style=for-the-badge)](#)

</div>

---

## ✨ Overview

This project enables you to recognize the 26 letters of the British Sign Language alphabet in real time using a webcam. It includes steps for dataset collection, hand landmark extraction, model training, and live prediction through a webcam stream. Perfect for computer vision and sign language beginners!

---

## 🎯 Features

- 🎥 **Real-time detection:** Predicts A–Z letter signs live via webcam.
- 🖐️ **MediaPipe Hand Tracking:** Detects 21 hand landmarks per frame.
- 🤖 **Machine Learning Powered:** Trained using a Random Forest classifier.
- 🔄 **End-to-end pipeline:** From dataset collection to real-time inference.
- 🗃️ **Auto-preprocessing:** Handles inconsistent feature vector lengths via padding/trimming.

---

## 🛠️ Tech Stack

- 🐍 Python 3
- 🎞️ OpenCV: For camera feed and image processing
- 🖐️ MediaPipe: For hand landmark extraction
- 🤖 Scikit-Learn: For model training (Random Forest)
- 🔢 NumPy: Data handling
- 📊 Matplotlib: Visualization

---

## ✨ Screenshots

<img width="1158" height="790" alt="Screenshot 2025-11-03 at 01 12 25" src="https://github.com/user-attachments/assets/0355f9b5-f625-4749-8404-5d7841a74680" />
<img width="1158" height="790" alt="Screenshot 2025-11-03 at 01 13 11" src="https://github.com/user-attachments/assets/7fe268c2-9e87-4bca-876d-85a3673c868c" />
<img width="1158" height="790" alt="Screenshot 2025-11-03 at 01 14 14" src="https://github.com/user-attachments/assets/612ee14d-f920-4eab-b724-8ad71a0ae030" />


---

## 🚀 Quick Start

Clone the repository and install dependencies:

```
git clone https://github.com/sgsjha/british-sign-language-detector-python.git
pip install opencv-python mediapipe scikit-learn
```

---

## 🧪 Usage

### 1. 📸 Collect Image Data

Run the script and press \`Q\` to begin capturing 100 samples per letter:


python collect_images.py

Images are saved in \`./data/{class}/{index}.jpg\`

---

### 2. 🧠 Process Dataset

Convert the captured images into landmark-based feature vectors:

python create_dataset.py

This creates \`data.pickle\` with data and labels.

---

### 3. 🏋️ Train the Classifier

Train a RandomForestClassifier model using Scikit-Learn:

python train_classifier.py

Saves the model to \`model.p\` and reports accuracy.

---

### 4. 🔮 Run Live Prediction

Launch the real-time webcam detector:

python inference_classifier.py

Predicted letter is overlaid on the webcam feed.

---

## 📂 Project Structure

```
.
├── collect_images.py          # Data collection via webcam
├── create_dataset.py          # Data processing
├── train_classifier.py        # Train Random Forest ML model
├── inference_classifier.py    # Real-time BSL letter inference
├── data/                      # Auto-generated image dataset
├── data.pickle                # Landmark-based dataset
├── model.p                    # Trained classifier
```
---

## 🧠 How It Works

- **Step 1:** Use \`cv2.VideoCapture\` to collect frames for each letter.
- **Step 2:** Feed images into MediaPipe Hands to extract landmark coordinates.
- **Step 3:** Flatten 21 landmarks × 2 (x, y) into a 42-length feature vector.
- **Step 4:** Use a Random Forest to classify vectors into labels 0–25 → A–Z.
- **Step 5:** Use OpenCV to stream webcam and overlay predictions in real time.

---

## 📬 Connect

- 👨‍💻 GitHub: https://github.com/sgsjha
- 🔗 LinkedIn: https://www.linkedin.com/in/sarthak-jhaa/
- 🌐 Portfolio: https://www.sarthakjha.dev/

---

Made with ❤️ to support inclusive communication through tech. 🙌
