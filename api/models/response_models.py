from pydantic import BaseModel
from typing import List, Optional

class ImageCaptionResponse(BaseModel):
    description: str
    tags: List[str]
    embedding: List[float]

class BatchImageCaptionResponse(BaseModel):
    results: List[ImageCaptionResponse]
    total_processed: int

class EmbeddingResponse(BaseModel):
    embedding: List[float]

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
