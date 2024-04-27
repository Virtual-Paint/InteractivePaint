from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio

from WebSocket.connection_manager import ConnectionManager
from ImageProcessing.image_processing import ImageProcessing
from ImageProcessing.GAN.inpainter import Inpainter
from models import InpaintModel


app = FastAPI()

origins = [
    'http://localhost:3000', 'http://192.168.0.178:3000'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)


manager = ConnectionManager()
inpainter = Inpainter()


@app.get('/')
async def root():
    return {'Wiadomość': ""}


@app.websocket('/virtual_paint')
async def virtual_paint(websocket: WebSocket):
    await manager.connect(websocket)
    recognizer = None

    try:
        while True:
            if not recognizer:
                recognizer = ImageProcessing()
            
            data = await websocket.receive_text()
            try:
                processed_image = recognizer.process_image(data)
                await manager.send_personal_message(processed_image, websocket)
            except:
                await manager.send_personal_message("Error", websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.post('/fill_sketch', 
          responses={200: {'content': {'image/png': {}}}}, 
          response_class=Response)
async def fill_sketch(body: InpaintModel):
    inpainted = inpainter.process_sketch(body)
    return JSONResponse(content={'inpainted': inpainted})
