from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from ultralytics import YOLO
import cv2
import numpy as np
from fastapi.responses import JSONResponse
import base64
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, adjust as necessary for security
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Load YOLO model (best.pt or last.pt)
model = YOLO("best_v2_augdata.pt")  # Replace with the path to your best.pt or last.pt

# Helper function to encode image to base64
def encode_image_to_base64(img):
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    # Read the uploaded image
    image_data = await file.read()
    img_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Run inference with YOLO model
    results = model(img)

    # Get the names of detected objects
    object_names = []
    for obj in results[0].boxes:
        if obj.cls is not None:
            object_names.append(results[0].names[int(obj.cls[0])])

    # If no objects were detected, set a default message
    if not object_names:
        object_names.append("No foreign debris")

    # Annotate the image
    annotated_img = results[0].plot()

    # Convert the annotated image to base64
    image_base64 = encode_image_to_base64(annotated_img)

    # Return the detected objects and the image
    return JSONResponse(content={
        "detected_objects": object_names,
        "annotated_image": image_base64
    })
