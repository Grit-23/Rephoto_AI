from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
from typing import Dict, Any
from utils.image_utils import get_qwen_service

router = APIRouter(prefix="/caption", tags=["caption"])

@router.post("/generate")
async def generate_image_caption(
    file: UploadFile = File(...)
) -> JSONResponse:

    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, 
                detail="이미지 파일만 업로드 가능합니다"
            )
        
        supported_formats = ["image/jpeg", "image/jpg", "image/png"]
        if file.content_type not in supported_formats:
            raise HTTPException(
                status_code=400,
                detail=f"지원 형식: {', '.join(supported_formats)}"
            )
        
        image_data = await file.read()
        
        try:
            image = Image.open(io.BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"이미지 오류: {str(e)}"
            )
        
        qwen_service = get_qwen_service()
        result = qwen_service.generate_caption_and_tags(image)
        
        response_data = {
            "explanation": result["explanation"],
            "tags": result["tags"],
            "explanation_embedding": result["explanation_embedding"]
        }
        
        return JSONResponse(
            status_code=200,
            content=response_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"오류: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"서버 오류: {str(e)}"
        )
