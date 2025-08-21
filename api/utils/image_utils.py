from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
import gc
import os
from PIL import Image
from typing import Dict, List, Any, Optional
from .embedding_utils import get_embedding_service
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QwenCaptionService:
    """Qwen Vision-Language 모델 서비스"""
    
    def __init__(self):
        self.model: Optional[Qwen2_5_VLForConditionalGeneration] = None
        self.processor: Optional[AutoProcessor] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()
    
    def _load_model(self) -> None:
        """모델 및 프로세서 로드"""
        try:
            # 기존 인스턴스 정리
            self._cleanup_resources()
            
            # CUDA 설정
            if self.device == "cuda":
                torch.cuda.empty_cache()
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            
            # 모델 로드
            logger.info("Qwen 모델 로딩 중...")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-3B-Instruct",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            # 프로세서 로드
            self.processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2.5-VL-3B-Instruct",
                trust_remote_code=True
            )
            
            self.model.eval()
            logger.info("모델 로드 완료")
            
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            raise RuntimeError(f"모델 로드 실패: {e}")
    
    def _cleanup_resources(self) -> None:
        """리소스 정리"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
    
    def unload(self) -> bool:
        """모델 언로드"""
        try:
            self._cleanup_resources()
            logger.info("모델 언로드 완료")
            return True
        except Exception as e:
            logger.error(f"언로드 실패: {e}")
            return False

    def reload(self) -> bool:
        """모델 재로드"""
        try:
            self._load_model()
            return True
        except Exception:
            return False
    
    def generate_caption_and_tags(self, image: Image.Image) -> Dict[str, Any]:
        """이미지에서 캡션과 태그 생성
        
        Args:
            image: PIL Image 객체
            
        Returns:
            설명, 태그, 개인정보 여부, 임베딩을 포함한 딕셔너리
        """
        try:
            # 모델 체크
            if self.model is None or self.processor is None:
                logger.info("모델 재로드 필요")
                self._load_model()

            # 이미지 전처리
            image = self._preprocess_image(image)

            # 프롬프트 생성
            prompt = self._create_prompt()
            
            # 입력 준비
            inputs = self._prepare_inputs(image, prompt)

            # 텍스트 생성
            output_text = self._generate_text(inputs)

            # 결과 파싱
            result = self._parse_output(output_text)
            
            # 임베딩 추가
            result = self._add_embedding(result)
            
            # 메모리 정리
            self._cleanup_after_generation()
            
            return result
            
        except Exception as e:
            logger.error(f"캡션 생성 실패: {e}")
            self._cleanup_after_generation()
            raise RuntimeError(f"캡션 생성 실패: {e}")
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """이미지 전처리"""
        try:
            image.load()
        except Exception as e:
            raise ValueError(f"이미지 로드 실패: {e}")
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        return image
    
    def _create_prompt(self) -> str:
        """프롬프트 생성"""
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": (
                    "이미지를 한 줄로 설명하세요.\n"
                    "태그: 핵심 단어 3개 이하, 쉼표로 구분\n"
                    "개인정보: YES 또는 NO\n\n"
                    "형식:\n"
                    "설명 한 줄\n"
                    "태그: 태그1, 태그2, 태그3\n"
                    "개인정보: NO"
                )}
            ]
        }]
        
        return self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    
    def _prepare_inputs(self, image: Image.Image, prompt: str) -> Dict:
        """모델 입력 준비"""
        inputs = self.processor(
            text=[prompt],
            images=[image],
            return_tensors="pt"
        )
        
        return inputs.to(self.device)
    
    def _generate_text(self, inputs: Dict) -> str:
        """텍스트 생성"""
        with torch.no_grad():
            # 생성 파라미터
            generation_config = {
                "max_new_tokens": 150,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "repetition_penalty": 1.1,
            }
            
            try:
                # 생성
                outputs = self.model.generate(
                    **inputs,
                    **generation_config
                )
                
                # 입력 부분 제거
                input_len = inputs.input_ids.shape[1]
                generated_ids = outputs[:, input_len:]
                
                # 디코딩 (None 값 방어)
                decoded = self.processor.tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
                
                # 결과 검증
                if decoded and decoded[0]:
                    return decoded[0]
                else:
                    raise ValueError("빈 출력")
                    
            except Exception as e:
                logger.warning(f"생성 실패, 폴백 모드: {e}")
                # 폴백: 더 보수적인 설정
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,  # greedy
                    repetition_penalty=1.2
                )
                
                input_len = inputs.input_ids.shape[1]
                generated_ids = outputs[:, input_len:]
                
                decoded = self.processor.tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
                
                if decoded and decoded[0]:
                    return decoded[0]
                else:
                    return "이미지 분석 중 오류가 발생했습니다.\n태그: 이미지, 분석, 오류\n개인정보: NO"
    
    def _parse_output(self, text: str) -> Dict[str, Any]:
        """출력 파싱"""
        result = {
            "explanation": "",
            "tags": [],
            "private_info": False
        }
        
        try:
            lines = text.strip().split('\n')
            
            # 첫 줄은 설명
            if lines:
                result["explanation"] = lines[0].strip()
            
            # 나머지 줄에서 태그와 개인정보 추출
            for line in lines[1:]:
                line = line.strip()
                if line.startswith('태그:') or line.lower().startswith('tags:'):
                    tag_text = line.split(':', 1)[1].strip()
                    tags = [t.strip() for t in tag_text.split(',') if t.strip()]
                    result["tags"] = tags[:3]  # 최대 3개
                elif line.startswith('개인정보:') or line.lower().startswith('privacy:'):
                    privacy_text = line.split(':', 1)[1].strip().upper()
                    result["private_info"] = 'YES' in privacy_text
            
            # 기본값 설정
            if not result["explanation"]:
                result["explanation"] = "이미지 분석 완료"
            
        except Exception as e:
            logger.warning(f"파싱 오류: {e}")
            result["explanation"] = text.strip() if text.strip() else "이미지 분석 완료"
        
        return result
    
    def _add_embedding(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """임베딩 추가"""
        try:
            embedding_service = get_embedding_service()
            result["explanation_embedding"] = embedding_service.encode_text(
                result["explanation"]
            )
        except Exception as e:
            logger.warning(f"임베딩 생성 실패: {e}")
            result["explanation_embedding"] = [0.0] * 1024  # 기본값
        
        return result
    
    def _cleanup_after_generation(self) -> None:
        """생성 후 메모리 정리"""
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            # 메모리 상태 로깅
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                logger.debug(f"GPU 메모리 사용: {allocated:.2f}GB")

# 싱글톤 인스턴스
_qwen_service: Optional[QwenCaptionService] = None

def get_qwen_service() -> QwenCaptionService:
    """Qwen 서비스 싱글톤 인스턴스 반환"""
    global _qwen_service
    if _qwen_service is None:
        _qwen_service = QwenCaptionService()
    return _qwen_service

def clear_qwen_service() -> None:
    """Qwen 서비스 정리 및 메모리 해제"""
    global _qwen_service
    if _qwen_service is not None:
        try:
            _qwen_service.unload()
        except Exception as e:
            logger.error(f"서비스 언로드 실패: {e}")
        finally:
            _qwen_service = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Qwen 서비스 메모리 정리 완료")
