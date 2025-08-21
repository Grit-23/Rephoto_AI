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
        self._load_model()
    
    def _load_model(self):
        try:
            torch.cuda.empty_cache()
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-3B-Instruct",
                torch_dtype=torch.float16,
                device_map="cuda",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
            
        except Exception as e:
            print(f"모델 로드 오류: {str(e)}")
            raise e
    
    # 모델 unload / VRAM 해제
    def unload(self) -> bool:
        try:
            if self.model is not None:
                del self.model
            if self.processor is not None:
                del self.processor
            self.model = None
            self.processor = None
            torch.cuda.empty_cache()
            gc.collect()
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
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
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
            
            return result
            
        except Exception as e:
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