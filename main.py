from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio

from WebSocket.connection_manager import ConnectionManager
from ImageProcessing.image_processing import ImageProcessing
from ImageProcessing.sketch_data import Sketch
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
image_processor = ImageProcessing()


@app.get('/')
async def root():
    return {'Wiadomość': ""}


@app.websocket('/virtual_paint')
async def virtual_paint(websocket: WebSocket):
    await manager.connect(websocket)
    sketch = None

    try:
        while True:
            if not sketch:
                sketch = Sketch(image_processor.kalman)
            
            data = await websocket.receive_json()
            try:
                if 'image' in data:
                    processed_image = image_processor.process_image(data.get('image'), sketch)
                    await manager.send_personal_message(processed_image, websocket)
                else:
                    sketch.set_settings(data)
            except:
                await manager.send_personal_message("Error", websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.post('/fill_sketch', 
          responses={200: {'content': {'image/png': {}}}}, 
          response_class=Response)
async def fill_sketch(body: InpaintModel):
    inpainted = image_processor.inpaint_sketch(body)
    return JSONResponse(content={'inpainted': inpainted})
