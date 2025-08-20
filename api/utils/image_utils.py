from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import gc
import os
import re
from PIL import Image
from typing import Dict, List, Any
from .embedding_utils import get_embedding_service

class QwenCaptionService:
    def __init__(self):
        self.model = None
        self.processor = None
        self.auto_unload = True  # 매 요청마다 자동 언로드
        self._load_model()
    
    def _load_model(self):
        try:
            # 강력한 메모리 정리
            torch.cuda.empty_cache()
            gc.collect()
            
            # CUDA 설정
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            
            # 모델 로드 전 추가 정리
            if hasattr(self, 'model') and self.model is not None:
                del self.model
            if hasattr(self, 'processor') and self.processor is not None:
                del self.processor
            
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-3B-Instruct",
                torch_dtype=torch.float16,
                device_map="cuda",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                attn_implementation="flash_attention_2"  # 더 안정적인 어텐션 구현
            )
            
            self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
            
            # 모델을 eval 모드로 설정
            self.model.eval()
            
            print("모델 로드 완료")
            
        except Exception as e:
            print(f"모델 로드 오류: {str(e)}")
            raise e
    
    # 모델 unload / VRAM 해제
    def unload(self) -> bool:
        try:
            if self.model is not None:
                # GPU에서 직접 삭제 (CPU로 이동하지 않음)
                del self.model
            if self.processor is not None:
                del self.processor
            self.model = None
            self.processor = None
            
            # 강력한 메모리 정리
            torch.cuda.empty_cache()
            gc.collect()
            
            print("모델 언로드 완료")
            return True
        except Exception as e:
            print(f"언로드 오류: {e}")
            return False

    # 모델 reload
    def reload(self) -> bool:
        ok = self.unload()
        if not ok: 
            return False
        self._load_model()
        return True

    # GPU 메모리 조회
    def gpu_memory(self) -> dict:
        try:
            free, total = torch.cuda.mem_get_info()
            return {"free": int(free), "total": int(total), "used": int(total - free)}
        except Exception as e:
            return {"error": str(e)}
    

    
    def generate_caption_and_tags(self, image: Image.Image) -> Dict[str, Any]:
        try:
            # 모델이 언로드되어 있으면 다시 로드
            if self.model is None or self.processor is None:
                print("모델 재로드 중...")
                # 기존 인스턴스 완전 정리
                self.unload()
                # 잠시 대기 후 재로드
                import time
                time.sleep(2)  # 더 긴 대기 시간
                self._load_model()
                
                # 모델 로드 후 추가 검증
                if self.model is None or self.processor is None:
                    raise RuntimeError("모델 로드 실패")
            
            torch.cuda.empty_cache()
            gc.collect()
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {
                            "type": "text", 
                            "text": """
                            이 이미지를 한 줄로 핵심만 뽑아 자세히 설명해주세요. 길이는 150 토큰 이하로 해주세요. 
                            그리고 이어서 '태그:'라고 쓰고 이 이미지를 나타내는 핵심 태그 3개 이하를 쉼표로 구분해서 작성해주세요.
                            
                            마지막으로 '개인정보:'라고 쓰고, 이 이미지에 개인정보(이름, 전화번호, 주민등록번호, 생년월일, 이메일, 신용카드번호, 계좌번호, 주소 등)가 포함되어 있으면 'YES', 없으면 'NO'라고 적어주세요.
                            """
                        },
                    ],
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            with torch.no_grad():
                # 더 안전한 생성 파라미터 설정
                generation_config = {
                    "max_new_tokens": 150,  # 토큰 수 줄임
                    "do_sample": False,     # 확률적 샘플링 비활성화
                    "num_beams": 1,         # 빔 서치 사용
                    "pad_token_id": self.processor.tokenizer.eos_token_id,
                    "eos_token_id": self.processor.tokenizer.eos_token_id,
                    "early_stopping": True,
                }
                
                # 입력 텐서 검증
                if torch.isnan(inputs.input_ids).any() or torch.isinf(inputs.input_ids).any():
                    raise ValueError("입력 텐서에 NaN 또는 Inf 값이 포함되어 있습니다")
                
                # 모델 상태 확인
                if not self.model.training:
                    self.model.eval()
                
                generated_ids = self.model.generate(
                    **inputs, 
                    **generation_config
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            if not output_text or not output_text[0]:
                raise Exception("캡션 생성 실패")
            
            result = self._parse_output(output_text[0])
            
            embedding_service = get_embedding_service()
            explanation_embedding = embedding_service.encode_text(result["explanation"])
            result["explanation_embedding"] = explanation_embedding
            
            # 요청 완료 후 강력한 메모리 정리
            if 'generated_ids' in locals():
                del generated_ids
            if 'generated_ids_trimmed' in locals():
                del generated_ids_trimmed
            if 'output_text' in locals():
                del output_text
            if 'inputs' in locals():
                del inputs
            if 'image_inputs' in locals():
                del image_inputs
            if 'video_inputs' in locals():
                del video_inputs
            if 'text' in locals():
                del text
            
            torch.cuda.empty_cache()
            gc.collect()
            
            # GPU 메모리 상태 확인
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"GPU 메모리 - 할당: {allocated:.2f}GB, 예약: {reserved:.2f}GB")
            
            # 매 요청마다 모델 언로드 (자원 제한으로 인해)
            if self.auto_unload:
                self.unload()
                print("모델이 언로드되었습니다. (GPU 메모리 해제)")
            
            return result
            
        except Exception as e:
            # 오류 발생 시에도 강력한 메모리 정리
            for var_name in ['generated_ids', 'generated_ids_trimmed', 'output_text', 'inputs', 'image_inputs', 'video_inputs', 'text']:
                if var_name in locals():
                    del locals()[var_name]
            
            torch.cuda.empty_cache()
            gc.collect()
            
            # 모델 상태 재설정
            if self.model is not None:
                try:
                    self.model.eval()
                except:
                    pass
            
            print(f"오류: {str(e)}")
            raise e
    
    def _parse_output(self, output: str) -> Dict[str, Any]:
        try:
            explanation = output.strip()
            tags = []
            private_info = False
            
            # 태그 파싱
            if '태그:' in output:
                parts = output.split('태그:', 1)
                explanation = parts[0].strip()
                remaining = parts[1].strip()
                
                # 개인정보 부분 확인
                if '개인정보:' in remaining:
                    tag_part, privacy_part = remaining.split('개인정보:', 1)
                    tag_part = tag_part.strip()
                    private_info = 'YES' in privacy_part.strip().upper()
                else:
                    tag_part = remaining
                
                # 태그 처리
                if tag_part:
                    raw_tags = [tag.strip() for tag in tag_part.split(',')]
                    tags = raw_tags[:3]
                    tags = [tag for tag in tags if tag]
            
            return {
                "explanation": explanation,
                "tags": tags,
                "private_info": private_info
            }
            
        except Exception as e:
            print(f"파싱 오류: {str(e)}")
            return {
                "explanation": output.strip(),
                "tags": [],
                "private_info": False
            }

_qwen_service = None

def get_qwen_service() -> QwenCaptionService:
    global _qwen_service
    if _qwen_service is None:
        _qwen_service = QwenCaptionService()
    return _qwen_service

def clear_qwen_service():
    """Qwen 서비스 인스턴스를 정리하고 메모리 해제"""
    global _qwen_service
    if _qwen_service is not None:
        try:
            # 서비스의 unload 메서드 사용
            _qwen_service.unload()
        except Exception as e:
            print(f"서비스 언로드 중 오류: {e}")
            # 강제 정리
            if hasattr(_qwen_service, 'model') and _qwen_service.model is not None:
                try:
                    del _qwen_service.model
                except:
                    pass
            if hasattr(_qwen_service, 'processor') and _qwen_service.processor is not None:
                try:
                    del _qwen_service.processor
                except:
                    pass
        
        del _qwen_service
        _qwen_service = None
        torch.cuda.empty_cache()
        gc.collect()
        print("Qwen 서비스 메모리 정리 완료")
