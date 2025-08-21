from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import embedding, caption

app = FastAPI(
    title="Gallery AI API",
    description="갤러리 이미지 분석 API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(embedding.router, prefix="/api/ai", tags=["embedding"])
app.include_router(caption.router, prefix="/api/ai", tags=["caption"])

@app.get("/")
async def root():
    return {"message": "Gallery AI API"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)