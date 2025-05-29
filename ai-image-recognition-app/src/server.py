from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from recognition.model import load_model, predict_image
import uvicorn
import os
import shutil

app = FastAPI()
app.mount("/static", StaticFiles(directory="static", html=True), name="static")
app.mount("/face_db", StaticFiles(directory="../face_db"), name="face_db")

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_DIR = os.path.join(os.path.dirname(__file__), "../face_db")

model = load_model()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    result = predict_image(contents, model)
    return JSONResponse(content=result)

@app.get("/db/list/")
async def list_people():
    people = []
    for filename in os.listdir(DB_DIR):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            people.append(os.path.splitext(filename)[0])
    return {"people": people}

@app.post("/db/add/")
async def add_person(name: str = Form(...), file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1]
    save_path = os.path.join(DB_DIR, f"{name}{ext}")
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    global model
    model = load_model()  # Reload model with new face
    return {"status": "success", "name": name}

@app.post("/db/delete/")
async def delete_person(name: str = Form(...)):
    deleted = False
    for filename in os.listdir(DB_DIR):
        if os.path.splitext(filename)[0] == name:
            os.remove(os.path.join(DB_DIR, filename))
            deleted = True
    global model
    model = load_model()  # Reload model after deletion
    return {"status": "deleted" if deleted else "not found", "name": name}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)