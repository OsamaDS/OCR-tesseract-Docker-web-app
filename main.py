from engine import OCR
from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
from starlette.requests import Request
import io
import re
import uvicorn
from pydantic import BaseModel



ocr = OCR()


app = FastAPI()

class ImageType(BaseModel):
 url: str

@app.get('/')
async def hello():
  return 'Hello world'

@app.post('/License_predict/')
def License_prediction(request: Request,
  file: bytes = File(...)):
  if request.method == 'POST':
    image_stream = io.BytesIO(file)
    image_stream.seek(0)
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    name_address, text = ocr.License_read_img(frame)
    return {"name_and_address":name_address, "text":text}
    return 'No post request found'

@app.post('/ID_predict/')
def ID_prediction(request: Request,
  file: bytes = File(...)):
  if request.method == 'POST':
    image_stream = io.BytesIO(file)
    image_stream.seek(0)
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    text = ocr.ID_read_img(frame)
    return {"text":text}
    return 'No post request found'

@app.post('/passport_prediction/')
def passport_prediction(request: Request,
  file: bytes = File(...)):
  if request.method == 'POST':
    image_stream = io.BytesIO(file)
    image_stream.seek(0)
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    text = ocr.passport_read_img(frame)
    return {"text":text}
    return 'No post request found'


if __name__=='__main__':
  uvicorn.run(app, host="0.0.0.0", port=5000)