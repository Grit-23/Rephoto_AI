from pydantic import BaseModel
from typing import List, Optional

class ImageCaptionResponse(BaseModel):
    explanation: str
    tags: List[str]
    explanation_embedding: List[float]
    private_info: bool
    
    class Config:
        json_schema_extra = {
            "example": {
                "explanation": "한 남성이 카메라를 향해 웃고 있는 프로필 사진",
                "tags": ["프로필", "남성", "웃음"],
                "explanation_embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                "private_info": False
            }
        }

class BatchImageCaptionResponse(BaseModel):
    results: List[ImageCaptionResponse]
    total_processed: int

class EmbeddingResponse(BaseModel):
    embedding: List[float]

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None