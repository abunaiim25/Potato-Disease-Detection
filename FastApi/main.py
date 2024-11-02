from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from io import BytesIO
import requests
from PIL import Image
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import keras
from keras.models import load_model



app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
MODEL = load_model("../saved_model/best_model2.keras")


#-----------------------------TEST START-------------------------------
@app.get("/")
def read_root():
    return {"message": "Welcome to the application...!"}

@app.get("/ping")
async def ping():
    try:
       import tensorflow as tf
       return(f"TensorFlow version: {tf.__version__}, Keras version: {keras. __version__}")
    except ImportError:
       return("TensorFlow is not installed.")
#-----------------------------TEST END-------------------------------

# Class name
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file_predict: UploadFile = File(...)
):
    # pass
    image = read_file_as_image(await file_predict.read())
    img_batch = np.expand_dims(image, 0)
    
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }




if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)