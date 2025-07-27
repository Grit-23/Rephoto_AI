from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import glob
import json
from PIL import Image
import torch
import gc

# 메모리 최적화 설정
torch.cuda.empty_cache()
gc.collect()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # GPU 2번 사용
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 디버깅을 위한 설정

print("공식 AWQ 모델 로딩 시작...")

# 공식 허깅페이스 문서에 따른 AWQ 모델 로딩
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct-AWQ", 
    torch_dtype="auto", 
    device_map="auto"
)

print("AWQ 모델 로딩 완료!")

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct-AWQ")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct-AWQ", min_pixels=min_pixels, max_pixels=max_pixels)

# 스크립트 파일의 위치를 기준으로 image 폴더 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
image_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))), "image")

if not os.path.exists(image_folder):
    print(f"Error: {image_folder} 폴더가 존재하지 않습니다.")
    exit(1)

jpg_files = glob.glob(os.path.join(image_folder, "*.jpg")) + glob.glob(os.path.join(image_folder, "*.JPG"))

if not jpg_files:
    print(f"Error: {image_folder} 폴더에 jpg 파일이 없습니다.")
    exit(1)

print(f"찾은 이미지 파일: {len(jpg_files)}개")

# 결과를 저장할 리스트
results = []

# 각 이미지에 대해 설명 생성
for i, image_path in enumerate(jpg_files):
    print(f"\n처리 중: {os.path.basename(image_path)} ({i+1}/{len(jpg_files)})")
    
    try:
        # 메모리 정리
        torch.cuda.empty_cache()
        gc.collect()
        
        # 이미지 로드
        raw_image = Image.open(image_path)
        
        # 이미지 정보 출력
        print(f"이미지 크기: {raw_image.size}, 모드: {raw_image.mode}")
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"CUDA 메모리 - 할당: {memory_allocated:.2f}GB, 예약: {memory_reserved:.2f}GB")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": raw_image,
                    },
                    {"type": "text", "text": "이 이미지를 한 줄로 자세히 설명해주세요."},
                ],
            }
        ]

        # Preparation for inference (공식 문서 그대로)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output (공식 문서 그대로)
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # 결과를 딕셔너리에 저장
        result = {
            "image_index": i,
            "image_filename": os.path.basename(image_path),
            "image_path": image_path,
            "description": output_text[0] if output_text else "No description generated"
        }
        results.append(result)
        
        print(f"설명: {output_text[0] if output_text else 'No description generated'}")
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        error_result = {
            "image_index": i,
            "image_filename": os.path.basename(image_path),
            "image_path": image_path,
            "description": f"Error: {str(e)}"
        }
        results.append(error_result)

# JSON 파일로 저장
output_file = os.path.join(script_dir, "output_qwen_awq.json")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n모든 이미지 처리 완료!")
print(f"결과가 {output_file}에 저장되었습니다.")
print(f"총 {len(results)}개의 이미지 처리됨")

