from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from recognition.model import load_model, predict_image
import uvicorn
import os
import shutil
import uuid

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
    for name in os.listdir(DB_DIR):
        person_dir = os.path.join(DB_DIR, name)
        if os.path.isdir(person_dir):
            images = []
            for filename in os.listdir(person_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    images.append(f"{name}/{filename}")
            if images:
                people.append({"name": name, "images": images})
    return {"people": people}

@app.post("/db/add/")
async def add_person(name: str = Form(...), file: UploadFile = File(...)):
    person_dir = os.path.join(DB_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    ext = os.path.splitext(file.filename)[1]
    unique_id = uuid.uuid4().hex[:8]
    save_path = os.path.join(person_dir, f"{name}_{unique_id}{ext}")
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    global model
    model = load_model()  # Reload model with new face(s)
    return {"status": "success", "name": name}

@app.post("/db/delete/")
async def delete_person(name: str = Form(...)):
    person_dir = os.path.join(DB_DIR, name)
    deleted = False
    if os.path.isdir(person_dir):
        shutil.rmtree(person_dir)
        deleted = True
    global model
    model = load_model()  # Reload model after deletion
    return {"status": "deleted" if deleted else "not found", "name": name}

@app.post("/db/delete_image/")
async def delete_image(name: str = Form(...), filename: str = Form(...)):
    person_dir = os.path.join(DB_DIR, name)
    file_path = os.path.join(person_dir, filename)
    deleted = False
    if os.path.exists(file_path):
        os.remove(file_path)
        deleted = True
    global model
    model = load_model()
    return {"status": "deleted" if deleted else "not found", "name": name, "filename": filename}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)