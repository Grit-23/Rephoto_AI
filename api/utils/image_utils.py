from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers.generation.logits_process import (
    LogitsProcessorList,
    InfNanRemoveLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    TemperatureLogitsWarper,
    TopPLogitsWarper,
)
import torch
import contextlib
import gc
import os
import re
from PIL import Image
from typing import Dict, List, Any
from .embedding_utils import get_embedding_service

try:
    torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
except Exception:
    pass

class QwenCaptionService:
    def __init__(self):
        self.model = None
        self.processor = None
        self.auto_unload = False  # 상시 로드 유지 (요청마다 언로드 금지)
        self._load_model()
    
    def _load_model(self):
        try:
            # CUDA 메모리 할당 정책 (가능하면 Dockerfile ENV로 설정 권장)
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

            # 기존 인스턴스 정리
            if getattr(self, 'model', None) is not None:
                del self.model
            if getattr(self, 'processor', None) is not None:
                del self.processor

            os.makedirs("/tmp/qwen_offload", exist_ok=True)

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-3B-Instruct",
                torch_dtype=torch.float16,
                device_map="cuda",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                attn_implementation="sdpa",
                max_memory={0: "14500MiB"},          # T4 16GB 여유
                offload_folder="/tmp/qwen_offload",
            )

            # 안정성 위해 fast processor 비활성화
            self.processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2.5-VL-3B-Instruct",
                use_fast=False
            )

            self.model.eval()
            print("모델 로드 완료")
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

            gc.collect()
            torch.cuda.empty_cache()
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
                self.unload()
                import time
                time.sleep(1.0)
                self._load_model()
                if self.model is None or self.processor is None:
                    raise RuntimeError("모델 로드 실패")

            # 이미지 검증/정규화
            try:
                image.load()  # 디코딩 강제 → 손상 조기 검출
            except Exception as e:
                raise RuntimeError(f"이미지 디코딩 실패: {e}")
            if image.mode != "RGB":
                image = image.convert("RGB")

            # 지시문을 짧고 형식 고정으로
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text":
                         "이미지 한 줄 설명만.\n"
                         "태그: 최대 3개, 쉼표로.\n"
                         "개인정보: YES 또는 NO.\n"
                         "형식:\n"
                         "<설명 한 줄>\n태그: a, b, c\n개인정보: NO"}
                    ],
                }
            ]

            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # 가장 단순한 바인딩 경로 사용 (버전 궁합 이슈 방지)
            inputs = self.processor(
                text=[prompt],
                images=[image],
                padding=True,
                return_tensors="pt",
            ).to("cuda")

            with torch.inference_mode():
                amp_dtype = torch.float16 if torch.cuda.is_available() else None
                amp_ctx = torch.cuda.amp.autocast(dtype=amp_dtype) if amp_dtype else contextlib.nullcontext()
                with amp_ctx:
                    pad_id = getattr(self.processor.tokenizer, "pad_token_id", None)
                    eos_id = getattr(self.processor.tokenizer, "eos_token_id", None)
                    if eos_id is None:
                        raise RuntimeError("Tokenizer has no eos_token_id")
                    if pad_id is None:
                        pad_id = eos_id

                    # logits processor 체인: NaN/Inf 제거 + 반복 억제 + 샘플링
                    processors = LogitsProcessorList([
                        InfNanRemoveLogitsProcessor(),
                        RepetitionPenaltyLogitsProcessor(1.2),
                        NoRepeatNGramLogitsProcessor(6),
                        TemperatureLogitsWarper(0.7),
                        TopPLogitsWarper(0.9),
                    ])

                    # 입력 텐서 NaN/Inf 방어적 정리
                    for k, v in inputs.items():
                        if torch.is_tensor(v) and v.is_floating_point():
                            if torch.isnan(v).any() or torch.isinf(v).any():
                                inputs[k] = torch.nan_to_num(v, nan=0.0, posinf=1e4, neginf=-1e4)

                    generation_kwargs = {
                        "max_new_tokens": 150,
                        "min_new_tokens": 12,
                        "do_sample": True,
                        "num_beams": 1,
                        "pad_token_id": pad_id,
                        "eos_token_id": eos_id,
                        "use_cache": True,
                        "logits_processor": processors,
                    }

                    # 1차 시도 (샘플링)
                    try:
                        generated_ids = self.model.generate(**inputs, **generation_kwargs)
                    except RuntimeError as e:
                        msg = str(e).lower()
                        if ("inf" in msg) or ("nan" in msg) or ("probability tensor" in msg):
                            # 폴백: 그리디 + 반복억제만 (거의 500 방지)
                            processors_fallback = LogitsProcessorList([
                                InfNanRemoveLogitsProcessor(),
                                RepetitionPenaltyLogitsProcessor(1.2),
                                NoRepeatNGramLogitsProcessor(6),
                            ])
                            generated_ids = self.model.generate(
                                **inputs,
                                max_new_tokens=150,
                                do_sample=False,          # 그리디
                                num_beams=1,
                                pad_token_id=pad_id,
                                eos_token_id=eos_id,
                                use_cache=True,
                                logits_processor=processors_fallback,
                            )
                        else:
                            raise

            # 프롬프트 길이만큼 잘라서 디코딩
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            if not output_text or not output_text[0]:
                raise RuntimeError("캡션 생성 실패")

            result = self._parse_output(output_text[0])

            # 임베딩 생성
            embedding_service = get_embedding_service()
            explanation_embedding = embedding_service.encode_text(result["explanation"])
            result["explanation_embedding"] = explanation_embedding

            # GPU 메모리 상태 로그
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"GPU 메모리 - 할당: {allocated:.2f}GB, 예약: {reserved:.2f}GB")

            # auto_unload가 True일 경우에만 언로드
            if self.auto_unload:
                self.unload()
                print("모델이 언로드되었습니다. (GPU 메모리 해제)")

            return result

        except Exception as e:
            # 지역 변수 정리
            for var_name in ['generated_ids', 'generated_ids_trimmed', 'output_text', 'inputs', 'prompt']:
                if var_name in locals():
                    try:
                        del locals()[var_name]
                    except:
                        pass

            # 모델 상태 재설정
            try:
                if self.model is not None:
                    self.model.eval()
            except:
                pass

            print(f"오류: {str(e)}")
            raise e
    
    def _parse_output(self, output: str) -> Dict[str, Any]:
        try:
            text = output.strip()
            explanation = text
            tags: List[str] = []
            private_info = False

            # 소문자 버전
            lo = text.lower()

            # '태그:' 또는 'tags:' 구간 분리
            tag_key_pos = None
            if '태그:' in text:
                tag_key_pos = text.find('태그:')
            elif 'tags:' in lo:
                tag_key_pos = lo.find('tags:')

            if tag_key_pos is not None:
                explanation = text[:tag_key_pos].strip()
                remaining = text[tag_key_pos:].strip()

                # '개인정보:' / 'pii:' / 'privacy:' 분리
                privacy_pos = None
                for key in ['개인정보:', 'pii:', 'privacy:']:
                    pos = remaining.lower().find(key) if key != '개인정보:' else remaining.find(key)
                    if pos != -1:
                        privacy_pos = pos
                        break

                if privacy_pos is not None:
                    tag_part = remaining[:privacy_pos].split(':', 1)[-1].strip()
                    privacy_part = remaining[privacy_pos:].split(':', 1)[-1].strip()
                else:
                    tag_part = remaining.split(':', 1)[-1].strip()
                    privacy_part = ""

                if tag_part:
                    raw_tags = [t.strip() for t in tag_part.split(',')]
                    tags = [t for t in raw_tags if t][:3]

                if privacy_part:
                    up = privacy_part.upper()
                    private_info = ('YES' in up) or ('TRUE' in up)

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
            _qwen_service.unload()
        except Exception as e:
            print(f"서비스 언로드 중 오류: {e}")
            if getattr(_qwen_service, 'model', None) is not None:
                try: del _qwen_service.model
                except: pass
            if getattr(_qwen_service, 'processor', None) is not None:
                try: del _qwen_service.processor
                except: pass

        del _qwen_service
        _qwen_service = None
        gc.collect()
        torch.cuda.empty_cache()
        print("Qwen 서비스 메모리 정리 완료")
