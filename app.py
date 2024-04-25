from fastapi import FastAPI, File, UploadFile
import numpy as np
from plate import loadmodel,predictimg
import cv2
import easyocr
reader=easyocr.Reader(['en'])

allowed = ['GJ03ER0563']

app = FastAPI()
modelYolo,modelVgg=loadmodel()

@app.post('/upload')
async def upload_file(imageFile: UploadFile = File(...)):
    image = await imageFile.read()

    with open('tmp.jpg', 'wb') as file:
        file.write(image)
    
    #call model, get output
    pred=predictimg(modelYolo,modelVgg,'tmp.jpg')
    print('predicted<', pred, '>')
    if pred in allowed:
        print('\n\n\n\nVehicle',pred,'Allowed.')
        return 'cool'
    else:
        print('\n\n\n\nVehicle Not Recognized.')
        return 'no cool'