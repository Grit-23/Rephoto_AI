from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import io
import asyncio
import time
import logging
from typing import Optional
from utils.image_utils import get_qwen_service, clear_qwen_service
from models.response_models import ImageCaptionResponse

# 로깅 설정
logger = logging.getLogger(__name__)

# 동시 처리 제한 설정
request_semaphore = asyncio.Semaphore(1)  # 순차 처리
last_request_time: float = 0.0
MIN_REQUEST_INTERVAL: float = 0.5  # 최소 요청 간격 (초)

router = APIRouter(prefix="/caption", tags=["caption"])

@router.post("/generate", response_model=ImageCaptionResponse)
async def generate_image_caption(
    file: UploadFile = File(...)
) -> ImageCaptionResponse:
    """이미지 캡션 생성 API
    
    Args:
        file: 업로드된 이미지 파일
        
    Returns:
        이미지 설명, 태그, 개인정보 여부, 임베딩
    """
    global last_request_time
    
    # 순차 처리
    async with request_semaphore:
        # 요청 간격 체크
        current_time = time.time()
        time_since_last = current_time - last_request_time
        
        if time_since_last < MIN_REQUEST_INTERVAL:
            wait_time = MIN_REQUEST_INTERVAL - time_since_last
            logger.info(f"요청 간격 제한: {wait_time:.2f}초 대기")
            await asyncio.sleep(wait_time)
        
        last_request_time = time.time()
        
        try:
            # 이미지 검증 및 로드
            image = await _load_and_validate_image(file)
            
            # 캡션 생성
            logger.info(f"캡션 생성 시작: {file.filename}")
            qwen_service = get_qwen_service()
            result = qwen_service.generate_caption_and_tags(image)
            
            logger.info(f"캡션 생성 완료: {file.filename}")
            
            return ImageCaptionResponse(
                explanation=result["explanation"],
                tags=result["tags"],
                explanation_embedding=result["explanation_embedding"],
                private_info=result["private_info"]
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"캡션 생성 실패: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"서버 오류: {str(e)}"
            )

async def _load_and_validate_image(file: UploadFile) -> Image.Image:
    """이미지 파일 검증 및 로드
    
    Args:
        file: 업로드된 파일
        
    Returns:
        PIL Image 객체
        
    Raises:
        HTTPException: 이미지 검증 실패 시
    """
    # Content-Type 검증
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="이미지 파일만 업로드 가능합니다"
        )
    
    # 지원 형식 검증
    supported_formats = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
    if file.content_type not in supported_formats:
        raise HTTPException(
            status_code=400,
            detail=f"지원 형식: {', '.join(sorted(supported_formats))}"
        )
    
    # 파일 크기 제한 (10MB)
    MAX_FILE_SIZE = 10 * 1024 * 1024
    image_data = await file.read()
    
    if len(image_data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="파일 크기는 10MB 이하여야 합니다"
        )
    
    # 이미지 로드
    try:
        image = Image.open(io.BytesIO(image_data))
        image.verify()  # 이미지 무결성 검증
        image = Image.open(io.BytesIO(image_data))  # verify 후 다시 열기
        
        # RGB 변환
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
            
        return image
        
    except Exception as e:
        logger.error(f"이미지 로드 실패: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"이미지 파일 오류: {str(e)}"
        )

@router.post("/free-vram")
async def free_vram():
    """GPU 메모리 정리"""
    import torch
    import gc
    
    try:
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info()
            
            return {
                "status": "success",
                "memory": {
                    "free_mb": int(free / 1024**2),
                    "total_mb": int(total / 1024**2),
                    "used_mb": int((total - free) / 1024**2)
                }
            }
        else:
            return {"status": "success", "message": "GPU not available"}
            
    except Exception as e:
        logger.error(f"메모리 정리 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reload-model")
async def reload_model():
    """모델 재로드"""
    try:
        clear_qwen_service()
        logger.info("모델 재로드 완료")
        
        return {
            "status": "success",
            "message": "모델이 재로드되었습니다"
        }
        
    except Exception as e:
        logger.error(f"모델 재로드 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"모델 재로드 실패: {str(e)}"
        )

@router.post("/force-clear-memory")
async def force_clear_memory():
    """강제 메모리 정리 (모델 언로드 포함)"""
    import torch
    import gc
    
    try:
        # 서비스 정리
        clear_qwen_service()
        
        # 메모리 정리
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            
            logger.info(f"메모리 정리 완료 - 할당: {allocated:.0f}MB, 예약: {reserved:.0f}MB")
            
            return {
                "status": "success",
                "memory": {
                    "allocated_mb": int(allocated),
                    "reserved_mb": int(reserved)
                }
            }
        else:
            return {
                "status": "success",
                "message": "메모리 정리 완료 (CPU 모드)"
            }
            
    except Exception as e:
        logger.error(f"메모리 정리 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"메모리 정리 실패: {str(e)}"
        )

@router.get("/memory")
async def get_memory_status():
    """현재 메모리 상태 조회"""
    import torch
    
    try:
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            
            return {
                "status": "success",
                "gpu": {
                    "device": torch.cuda.get_device_name(),
                    "free_mb": int(free / 1024**2),
                    "total_mb": int(total / 1024**2),
                    "allocated_mb": int(allocated / 1024**2),
                    "reserved_mb": int(reserved / 1024**2),
                    "usage_percent": round((total - free) / total * 100, 1)
                }
            }
        else:
            return {
                "status": "success",
                "message": "GPU not available"
            }
            
    except Exception as e:
        logger.error(f"메모리 상태 조회 실패: {e}")
        return {
            "status": "error",
            "error": str(e)
        }