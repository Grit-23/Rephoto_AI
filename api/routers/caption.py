from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import io
from typing import Dict, Any
from utils.image_utils import get_qwen_service
from models.response_models import ImageCaptionResponse

router = APIRouter(prefix="/caption", tags=["caption"])

@router.post("/generate", response_model=ImageCaptionResponse)
async def generate_image_caption(
    file: UploadFile = File(...)
) -> ImageCaptionResponse:

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
        
        return ImageCaptionResponse(
            explanation=result["explanation"],
            tags=result["tags"],
            explanation_embedding=result["explanation_embedding"],
            private_info=result["private_info"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"오류: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"서버 오류: {str(e)}"
        )

# VRAM 비우기
@router.post("/free-vram")
async def free_vram():
    import torch, gc
    try:
        gc.collect()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        return {"status": "ok", "free": int(free), "total": int(total), "used": int(total - free)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 모델 unload
@router.post("/reload-model")
async def reload_model():
    svc = get_qwen_service()
    ok = svc.reload()
    if not ok:
        raise HTTPException(status_code=500, detail="모델 unload 실패")
    return {"status": "reloaded"}

# GPU 메모리 조회
@router.get("/memory")
async def gpu_memory():
    svc = get_qwen_service()
    return svc.gpu_memory()