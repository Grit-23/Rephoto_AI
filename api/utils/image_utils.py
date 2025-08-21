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
            # 생성 전 메모리 정리
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
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
                    "이 이미지를 분석해주세요. 다음 형식으로 답변하세요:\n\n"
                    "한 줄로 이미지의 주요 내용을 설명\n"
                    "태그: 관련 키워드 1-3개\n"
                    "개인정보: YES 또는 NO"
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
        # 프로세서 호출 전 이미지 크기 체크
        max_size = 1920
        if max(image.size) > max_size:
            # 이미지가 너무 크면 리사이즈
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            logger.debug(f"이미지 리사이즈: {image.size}")
        
        inputs = self.processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
            padding=True
        )
        
        # 디바이스로 이동
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v 
                  for k, v in inputs.items()}
        
        return inputs
    
    def _generate_text(self, inputs: Dict) -> str:
        """텍스트 생성"""
        with torch.no_grad():
            try:
                # 모델을 eval 모드로 확실히 설정
                self.model.eval()
                
                # 안전한 생성 파라미터 (Qwen 모델에 맞게 조정)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    min_new_tokens=10,
                    do_sample=False,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
                
                # 입력 부분 제거 (inputs는 딕셔너리)
                input_len = inputs['input_ids'].shape[1]
                generated_ids = outputs[:, input_len:]
                
                # 생성된 토큰이 있는지 확인
                if generated_ids.shape[1] == 0:
                    raise ValueError("No tokens generated")
                
                # 디코딩
                text = self.processor.tokenizer.decode(
                    generated_ids[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
                
                # 결과 검증 및 정리
                if text and len(text.strip()) > 0:
                    # 강력한 특수 문자 패턴 제거
                    import re
                    # 특수 문자 반복 제거 (!!!!! 같은 경우)
                    text = re.sub(r'[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>\/?]{3,}', '', text)
                    # 같은 문자 3번 이상 반복 제거
                    text = re.sub(r'(.)\1{3,}', r'\1\1', text)
                    # 연속된 공백 제거
                    text = re.sub(r'\s+', ' ', text)
                    
                    # 최소 길이 체크
                    cleaned_text = text.strip()
                    if len(cleaned_text) >= 5:
                        return cleaned_text
                    else:
                        raise ValueError("Generated text too short")
                else:
                    raise ValueError("Empty output")
                    
            except Exception as e:
                logger.warning(f"생성 실패, 폴백 모드: {e}")
                
                # 모델 상태 리셋
                self.model.eval()
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
                try:
                    # 폴백: 매우 보수적인 설정
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=80,
                        min_new_tokens=20,
                        do_sample=False,  # greedy decoding
                        repetition_penalty=1.3,
                        no_repeat_ngram_size=4,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                    )
                    
                    input_len = inputs['input_ids'].shape[1]
                    generated_ids = outputs[:, input_len:]
                    
                    if generated_ids.shape[1] > 0:
                        text = self.processor.tokenizer.decode(
                            generated_ids[0],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False
                        )
                        
                        if text and len(text.strip()) > 0:
                            # 강력한 특수 문자 필터링
                            import re
                            text = re.sub(r'[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>\/?]{3,}', '', text)
                            text = re.sub(r'(.)\1{3,}', r'\1\1', text)
                            text = re.sub(r'\s+', ' ', text)
                            
                            cleaned_text = text.strip()
                            if len(cleaned_text) >= 5:
                                return cleaned_text
                    
                except Exception as e2:
                    logger.error(f"폴백도 실패: {e2}")
                
                # 최종 폴백 메시지
                return "이미지를 분석했습니다.\n태그: 이미지, 사진, 파일\n개인정보: NO"
    
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
        # 모델 상태 리셋
        if self.model is not None:
            self.model.eval()
        
        # 메모리 정리
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # GPU 작업 완료 대기
            
            # 메모리 상태 로깅
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logger.debug(f"GPU 메모리 - 할당: {allocated:.2f}GB, 예약: {reserved:.2f}GB")

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
