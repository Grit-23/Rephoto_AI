from pydantic import BaseModel
from typing import List

class ImageCaptionRequest(BaseModel):
    image_base64: str

class BatchImageCaptionRequest(BaseModel):
    images: List[str]

class EmbeddingRequest(BaseModel):
    text: str
