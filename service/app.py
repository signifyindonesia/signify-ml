# File: app.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from utils.image_inference import predict_from_image_file
from utils.video_inference import predict_from_video_file

from fastapi import WebSocket, WebSocketDisconnect
from utils.realtime_inference import load_realtime_resources, predict_frame
import uvicorn

app = FastAPI(title="Signify ML Service")

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="File harus berupa gambar (jpg/png)")

    prediction = await predict_from_image_file(file)
    return JSONResponse(content=prediction)


@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    if file.content_type not in ["video/mp4", "video/avi"]:
        raise HTTPException(status_code=400, detail="File harus berupa video (mp4/avi)")

    prediction = await predict_from_video_file(file)
    return JSONResponse(content=prediction)

# Load model & resources for realtime inference
realtime_model, realtime_labels, realtime_detector = load_realtime_resources()

@app.websocket("/ws/realtime")
async def websocket_realtime(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            label, confidence = predict_frame(frame, realtime_model, realtime_labels, realtime_detector)

            await websocket.send_json({
                "prediction": label,
                "confidence": confidence
            })
    except WebSocketDisconnect:
        print("Client disconnected from realtime websocket")
    except Exception as e:
        await websocket.send_json({"error": str(e)})

# Jalankan dengan: uvicorn app:app --reload
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
