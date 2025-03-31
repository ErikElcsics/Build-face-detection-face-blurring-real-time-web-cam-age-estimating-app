import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1

# Load Face Detection Model (MTCNN)
mtcnn = MTCNN(keep_all=True)

# Load Pre-trained Age & Gender Model (Optional)
age_gender_model = InceptionResnetV1(pretrained='vggface2').eval()

# Streamlit UI
st.set_page_config(page_title="Face Detection, Face Blurring, Real-time Web Cam App and Age Estimating (starter) App", layout="wide")
st.title("Advanced Face Detection App")
st.write("Upload an image, and the app will detect faces, predict age/gender, and allow face blurring.")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Checkbox for Face Blurring
blur_faces = st.checkbox("Blur Detected Faces for Privacy")

if uploaded_file:
    # Convert file to OpenCV image
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Detect faces
    boxes, _ = mtcnn.detect(image_np)

    if boxes is not None:
        for (x1, y1, x2, y2) in boxes.astype(int):
            # Blur face if option is enabled
            if blur_faces:
                face_roi = image_np[y1:y2, x1:x2]
                face_roi = cv2.GaussianBlur(face_roi, (99, 99), 30)
                image_np[y1:y2, x1:x2] = face_roi
            else:
                cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Age & Gender Prediction
            face_crop = Image.fromarray(image_np[y1:y2, x1:x2])
            transform = transforms.Compose([transforms.Resize((160, 160)), transforms.ToTensor()])
            face_tensor = transform(face_crop).unsqueeze(0)

            with torch.no_grad():
                embeddings = age_gender_model(face_tensor)
                # age = int(embeddings.mean().item() * 100)  # Fake age estimation for now
                age = int((embeddings.mean().item() + 1) * 50)  # Scale to a reasonable range

            cv2.putText(image_np, f"Age: {age}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert back to PIL Image for Streamlit display
    detected_image = Image.fromarray(image_np)
    st.image(detected_image, caption=f"Detected Faces: {len(boxes)}", use_container_width=True)

    if len(boxes) == 0:
        st.warning("No faces detected.")

# Real-Time Webcam Face Detection
if st.button("Start Webcam Face Detection"):
    st.write("Press 'Q' to exit webcam mode.")
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces in webcam feed
        boxes, _ = mtcnn.detect(frame)
        if boxes is not None:
            for (x1, y1, x2, y2) in boxes.astype(int):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        cv2.imshow("Webcam Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
