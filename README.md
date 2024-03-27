# Face-detection-and-sentiment-analysis

# FastAPI Video Processing Service

This is a FastAPI-based web service for processing videos to detect faces and analyze sentiment using DeepFace.

## Setup

### Installation

1. Clone the repository:

    ```bash
    git clone <repository-url>
    ```

2. Navigate to the project directory:

    ```bash
    cd fastapi-video-processing-service
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Usage

1. Run the FastAPI server:

    ```bash
    uvicorn main:app --reload
    ```

2. Access the FastAPI interactive documentation at `http://127.0.0.1:8000/docs` in your web browser.

3. Use the `/process-video/` endpoint to upload a video file from Google Drive and initiate the processing.

4. Monitor the progress and results through the console and the output video generated.

## Endpoints

- **POST /process-video/**
  - Upload a video file from Google Drive to perform face detection and sentiment analysis.

## Technologies Used

- FastAPI
- OpenCV
- NumPy
- Requests
- DeepFace
- MTCNN

## Contributors

- [Your Name](mailto:your.email@example.com)

## License

This project is licensed under the [MIT License](LICENSE).
