import torch
from transformers import AutoTokenizer
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
from llava.mm_utils import tokenizer_image_token, process_images
import requests
from PIL import Image
import os
import glob
import json

model_name = "NCSOFT/VARCO-VISION-14B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlavaQwenForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
    low_cpu_mem_usage=True,
    device_map="auto"
)

vision_tower = model.get_vision_tower()
image_processor = vision_tower.image_processor

# 간단한 프롬프트 생성 (템플릿 오류 회피) - 한국어로 변경
prompt = "<|im_start|>user\n<image>\n이 이미지를 자세히 설명해주세요.<|im_end|>\n<|im_start|>assistant\n"

IMAGE_TOKEN_INDEX = -200
EOS_TOKEN = "<|im_end|>"
"""
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
input_ids = input_ids.unsqueeze(0).to(model.device)

image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
raw_image = Image.open(requests.get(image_url, stream=True).raw)
image_tensors = process_images([raw_image], image_processor, model.config)
image_tensors = [image_tensor.half().to(model.device) for image_tensor in image_tensors]
image_sizes = [raw_image.size]

with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=image_tensors,
        image_sizes=image_sizes,
        do_sample=False,
        max_new_tokens=1024,
        use_cache=True,
    )

outputs = tokenizer.batch_decode(output_ids)[0]
if outputs.endswith(EOS_TOKEN):
    outputs = outputs[: -len(EOS_TOKEN)]

outputs = outputs.strip()
print(outputs)
"""
# 스크립트 파일의 위치를 기준으로 image 폴더 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
image_folder = os.path.join(os.path.dirname(script_dir), "image")

if not os.path.exists(image_folder):
    print(f"Error: {image_folder} 폴더가 존재하지 않습니다.")
    print(f"현재 스크립트 위치: {script_dir}")
    print(f"찾고 있는 이미지 폴더: {image_folder}")
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
    print(f"\n{'='*50}")
    print(f"처리 중: {os.path.basename(image_path)} ({i+1}/{len(jpg_files)})")
    print(f"{'='*50}")
    
    try:
        # 이미지 로드
        raw_image = Image.open(image_path)
        
        # 프롬프트 생성
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).to(model.device)
        
        # 이미지 처리
        image_tensors = process_images([raw_image], image_processor, model.config)
        image_tensors = [image_tensor.half().to(model.device) for image_tensor in image_tensors]
        image_sizes = [raw_image.size]
        
        # 설명 생성
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensors,
                image_sizes=image_sizes,
                do_sample=False,
                max_new_tokens=1024,
                use_cache=True,
            )
        
        outputs = tokenizer.batch_decode(output_ids)[0]
        if outputs.endswith(EOS_TOKEN):
            outputs = outputs[: -len(EOS_TOKEN)]
        
        # 프롬프트 부분 제거하여 실제 생성된 텍스트만 추출
        outputs = outputs.strip()
        if prompt in outputs:
            outputs = outputs.replace(prompt, "").strip()
        
        # 결과를 딕셔너리에 저장
        result = {
            "image_index": i,
            "image_filename": os.path.basename(image_path),
            "image_path": image_path,
            "description": outputs
        }
        results.append(result)
        
        print(f"이미지: {os.path.basename(image_path)}")
        print(f"설명: {outputs}")
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        # 오류가 발생한 경우에도 기록
        error_result = {
            "image_index": i,
            "image_filename": os.path.basename(image_path),
            "image_path": image_path,
            "description": f"Error: {str(e)}"
        }
        results.append(error_result)
        continue

# JSON 파일로 저장 - 스크립트와 같은 폴더에 저장
output_file = os.path.join(script_dir, "output_varco.json")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n모든 이미지 처리 완료!")
print(f"결과가 {output_file}에 저장되었습니다.")
print(f"총 {len(results)}개의 이미지 처리됨")
