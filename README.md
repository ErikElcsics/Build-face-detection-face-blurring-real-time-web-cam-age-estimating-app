# Build Face Detection, Face Blurring, Real-time Face Detection with Web Cam and Age Estimating App

This Streamlit-powered app detects faces in images and webcam feeds using MTCNN (Multi-task Cascaded Convolutional Networks). It also includes additional features such as age estimation, face blurring, and real-time face detection.

Features:
- Browse and Upload images (JPG, PNG, JPEG)
- Face Detection – Uses MTCNN to detect multiple faces in an image.
- Face Blurring – Provides an option to blur detected faces for privacy protection.
- Real-time Webcam Detection – Detects faces live via webcam.
- Age Estimation – Implements a pre-trained InceptionResnetV1 model for approximate age prediction.
	- Alternatively, consider replacing the InceptionResnetV1 model with a dedicated age estimation model, such as:
		- deepface.DeepFace.analyze - Check GITHub
		- fairface (PyTorch) - https://github.com/dchen236/FairFace
- Count of number of faces detected

Installation:

Clone the repository:
- git clone https://github.com/ErikElcsics/Build-face-detection-face-blurring-real-time-web-cam-age-estimating-app.git
- cd face-detection-app

Install dependencies:
- pip install -r requirements.txt

Uses libraries:
- streamlit
- opencv-python
- numpy
- pillow
- torch
- torchvision
- facenet-pytorch

Run the app:
- streamlit run FaceDetection_FaceBlurring_Real-time_Web_Cam_Age_EstimatingApp.py

Usage:
- Upload an image to detect faces.
- Enable "Blur Detected Faces" if privacy is needed.
- Click "Start Webcam Face Detection" to detect faces in real time.

Face Detection

![image](https://github.com/user-attachments/assets/035048c1-3720-4b0a-9f73-80a9e62e3e75)

Blur Detected Faces

![image](https://github.com/user-attachments/assets/bf121695-b05c-4ec8-8b2b-d81e0291146f)

Age Detection

![image](https://github.com/user-attachments/assets/8a61abe6-7c42-4e43-924c-13e659bdc359)

Real-Time Web Cam Face Detection

![image](https://github.com/user-attachments/assets/d3fdba2d-b057-421d-abd9-2de286c1f3c5)



