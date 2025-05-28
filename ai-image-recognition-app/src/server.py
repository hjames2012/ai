from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from recognition.model import load_model, predict_image
import uvicorn

app = FastAPI()
app.mount("/", StaticFiles(directory="static", html=True), name="static")

model = load_model()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    prediction = predict_image(contents, model)
    return JSONResponse(content={"prediction": prediction})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)