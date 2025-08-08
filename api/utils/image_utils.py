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
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
            
        except Exception as e:
            print(f"모델 로드 오류: {str(e)}")
            raise e
    
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
            if '태그:' in output:
                parts = output.split('태그:', 1)
                explanation = parts[0].strip()
                tag_part = parts[1].strip()
            else:
                explanation = output.strip()
                tag_part = ""
            
            tags = []
            if tag_part:
                raw_tags = [tag.strip() for tag in tag_part.split(',')]
                tags = raw_tags[:3]
                tags = [tag for tag in tags if tag]
            
            return {
                "explanation": explanation,
                "tags": tags
            }
            
        except Exception as e:
            print(f"파싱 오류: {str(e)}")
            return {
                "explanation": output.strip(),
                "tags": []
            }

_qwen_service = None

def get_qwen_service() -> QwenCaptionService:
    global _qwen_service
    if _qwen_service is None:
        _qwen_service = QwenCaptionService()
    return _qwen_service
