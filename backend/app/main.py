from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Backend is working!"}

@app.post("/analyze")
async def analyze_text(text: str = Form(...)):
    # Dummy NLP response
    return {
        "summary": "Sample summary.",
        "sentiment": "positive",
        "keywords": ["NLP", "text", "analyze"],
    }
