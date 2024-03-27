# Step 1: Import necessary libraries and install additional packages if needed
import cv2
import numpy as np
import requests
import tempfile
from deepface import DeepFace
from fastapi import FastAPI
from mtcnn import MTCNN

# Step 2: Define FastAPI app
app = FastAPI()

# Function to download the video from Google Drive link
def download_video_from_drive(gdrive_url):
    # Extract file ID from the Google Drive link
    file_id = gdrive_url.split("/")[-2]
    # Generate the download link
    download_url = f"https://drive.google.com/uc?id={file_id}"
    # Download the video file
    response = requests.get(download_url)
    # Save the video file to a temporary directory
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(response.content)
        return temp_file.name

# Initialize MTCNN for face detection
detector = MTCNN()
def detect_faces(frame):
    # Convert frame to RGB (MTCNN requires RGB format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces using MTCNN
    detections = detector.detect_faces(rgb_frame)

    # Initialize list to store detected faces
    detected_faces = []

    # Iterate over the detected faces
    for detection in detections:
        x, y, width, height = detection['box']
        confidence = detection['confidence']

        # Filter out low-confidence detections
        if confidence > 0.5:
            detected_faces.append((x, y, width, height))

    return detected_faces

# Route to process video
@app.post("/process-video/")
async def process_video(gdrive_url: str):
    # Download the video from the provided Google Drive link
    video_path = download_video_from_drive(gdrive_url)

    # Read the input video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video file"}

    # Initialize MTCNN for face detection
    detector = MTCNN()

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    output_video_path = "sentiment_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        return {"error": "Could not open VideoWriter"}

    detected_faces = {}
    next_face_id = 1

    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces in the frame
        faces = detect_faces(frame)
        print("Here")
        for (x, y, w, h) in faces:
            roi = frame[y:y+h, x:x+w]
            result = DeepFace.analyze(roi, actions=['emotion'], enforce_detection=False)
            if result:
                emotions = result[0].get('emotion', {})
                emotion = max(emotions, key=emotions.get)
            else:
                emotion = "Neutral"

            matched_face_id = None
            for fid, (prev_x, prev_y, prev_w, prev_h, _) in detected_faces.items():
                x_overlap = max(0, min(x + w, prev_x + prev_w) - max(x, prev_x))
                y_overlap = max(0, min(y + h, prev_y + prev_h) - max(y, prev_y))
                overlap_area = x_overlap * y_overlap
                area_ratio = overlap_area / min(w * h, prev_w * prev_h)
                if area_ratio > 0.5:
                    matched_face_id = fid
                    break

            if matched_face_id is not None:
                face_id = matched_face_id
                detected_faces[face_id] = (x, y, w, h, emotion)
            else:
                face_id = next_face_id
                detected_faces[face_id] = (x, y, w, h, emotion)
                next_face_id += 1

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Face {face_id}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return {"message": "Face detection and sentiment analysis completed. Output video saved successfully."}

# Step 3: Run FastAPI app using uvicorn
import uvicorn
uvicorn.run(app, host="127.0.0.1", port=8000)
