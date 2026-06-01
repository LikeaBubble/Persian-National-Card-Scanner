from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
from Pipeline import pipeline

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["scanner"] = pipeline()    
    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan)


@app.get("/")
def root():
    return {"message": "Server is alive"}

@app.post("/scan")
def scan_card(file: UploadFile = File(...)):
    
    image_bytes = file.file.read()
    result = ml_models['scanner'].execute(image_bytes)
    
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000)