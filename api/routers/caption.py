from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import io
import asyncio
import time
from typing import Dict, Any
from utils.image_utils import get_qwen_service, clear_qwen_service
from models.response_models import ImageCaptionResponse

# 요청 순차 처리용 세마포어
MAX_CONCURRENT_REQUESTS = 1  # 동시 요청 수 제한
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# 마지막 요청 시간 추적
last_request_time = 0
MIN_REQUEST_INTERVAL = 0.5  # 최소 요청 간격 (초)

router = APIRouter(prefix="/caption", tags=["caption"])

@router.post("/generate", response_model=ImageCaptionResponse)
async def generate_image_caption(
    file: UploadFile = File(...)
) -> ImageCaptionResponse:
    global last_request_time
    
    # 요청 간격 제한
    current_time = time.time()
    if current_time - last_request_time < MIN_REQUEST_INTERVAL:
        await asyncio.sleep(MIN_REQUEST_INTERVAL - (current_time - last_request_time))
    last_request_time = time.time()

    # 세마포어로 순차 처리
    async with request_semaphore:
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
    try:
        clear_qwen_service()
        return {"status": "reloaded", "message": "모델 메모리 정리 완료"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 unload 실패: {str(e)}")

# 강제 메모리 정리
@router.post("/force-clear-memory")
async def force_clear_memory():
    import torch
    import gc
    try:
        clear_qwen_service()
        torch.cuda.empty_cache()
        gc.collect()
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            return {
                "status": "cleared",
                "allocated_gb": round(allocated, 2),
                "reserved_gb": round(reserved, 2)
            }
        return {"status": "cleared", "message": "GPU 메모리 정리 완료"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"메모리 정리 실패: {str(e)}")

# GPU 메모리 조회
@router.get("/memory")
async def gpu_memory():
    svc = get_qwen_service()
    return svc.gpu_memory()