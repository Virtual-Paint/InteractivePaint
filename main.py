from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

from WebSocket.connection_manager import ConnectionManager
from GesturesRecognition.proces_input import InputProcessing

app = FastAPI()
recognizer = InputProcessing()

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

@app.get('/')
async def root():
    return {'Wiadomość': ""}


@app.post('/fill_sketch', 
          responses={200: {'content': {'image/png': {}}}}, 
          response_class=Response)
async def fill_sketch(sketch: UploadFile):
    image_bytes = []
    #TODO
    return Response(content=image_bytes, media_type='image/png')


@app.websocket('/virtual_paint')
async def virtual_paint(websocket: WebSocket):
    await manager.connect(websocket)

    try:
        while True:
            data = await websocket.receive_text()
            recognizer.bytes_to_image(data)
            await manager.send_personal_message(f'Odebrano', websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
