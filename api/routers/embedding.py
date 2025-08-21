from fastapi import APIRouter, HTTPException
from models.request_models import EmbeddingRequest
from models.response_models import EmbeddingResponse, ErrorResponse
from utils.embedding_utils import get_embedding_service

router = APIRouter(prefix="/embedding")

@router.post("/embed-text", response_model=EmbeddingResponse)
async def embed_text(request: EmbeddingRequest):
    try:
        embedding_service = get_embedding_service()
        
        embedding = embedding_service.encode_text(request.text)
        
        return EmbeddingResponse(embedding=embedding)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"임베딩 생성 중 오류가 발생했습니다: {str(e)}"
        )

@router.get("/health")
async def embedding_health_check():
    try:
        embedding_service = get_embedding_service()
        
        test_embedding = embedding_service.encode_text("테스트")
        
        return {
            "status": "healthy",
            "model": "e5-large-korean",
            "embedding_dimension": len(test_embedding)
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"임베딩 서비스가 정상적으로 작동하지 않습니다: {str(e)}"
        )