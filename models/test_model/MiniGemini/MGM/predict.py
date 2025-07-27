import torch

from mgm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mgm.conversation import conv_templates, SeparatorStyle
from mgm.model.builder import load_pretrained_model
from mgm.utils import disable_torch_init
from mgm.mm_utils import tokenizer_image_token
from transformers.generation.streamers import TextIteratorStreamer

from PIL import Image

import requests
from io import BytesIO

from cog import BasePredictor, Input, Path, ConcatenateIterator
import time
import subprocess
from threading import Thread

import os
os.environ["HUGGINGFACE_HUB_CACHE"] = os.getcwd() + "/weights"

# url for the weights mirror
REPLICATE_WEIGHTS_URL = "https://weights.replicate.delivery/default"
# files to download from the weights mirrors
weights = [
    {
        "dest": "liuhaotian/llava-v1.5-13b",
        # git commit hash from huggingface
        "src": "llava-v1.5-13b/006818fc465ebda4c003c0998674d9141d8d95f8",
        "files": [
            "config.json",
            "generation_config.json",
            "pytorch_model-00001-of-00003.bin",
            "pytorch_model-00002-of-00003.bin",
            "pytorch_model-00003-of-00003.bin",
            "pytorch_model.bin.index.json",
            "special_tokens_map.json",
            "tokenizer.model",
            "tokenizer_config.json",
        ]
    },
    {
        "dest": "openai/clip-vit-large-patch14-336",
        "src": "clip-vit-large-patch14-336/ce19dc912ca5cd21c8a653c79e251e808ccabcd1",
        "files": [
            "config.json",
            "preprocessor_config.json",
            "pytorch_model.bin"
        ],
    }
]

def download_json(url: str, dest: Path):
    res = requests.get(url, allow_redirects=True)
    if res.status_code == 200 and res.content:
        with dest.open("wb") as f:
            f.write(res.content)
    else:
        print(f"Failed to download {url}. Status code: {res.status_code}")

def download_weights(baseurl: str, basedest: str, files: list[str]):
    basedest = Path(basedest)
    start = time.time()
    print("downloading to: ", basedest)
    basedest.mkdir(parents=True, exist_ok=True)
    for f in files:
        dest = basedest / f
        url = os.path.join(REPLICATE_WEIGHTS_URL, baseurl, f)
        if not dest.exists():
            print("downloading url: ", url)
            if dest.suffix == ".json":
                download_json(url, dest)
            else:
                subprocess.check_call(["pget", url, str(dest)], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        for weight in weights:
            download_weights(weight["src"], weight["dest"], weight["files"])
        disable_torch_init()
    
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model("liuhaotian/llava-v1.5-13b", model_name="llava-v1.5-13b", model_base=None, load_8bit=False, load_4bit=False)

    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(description="Prompt to use for text generation"),
        top_p: float = Input(description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens", ge=0.0, le=1.0, default=1.0),
        temperature: float = Input(description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic", default=0.2, ge=0.0),
        max_tokens: int = Input(description="Maximum number of tokens to generate. A word is generally 2-3 tokens", default=1024, ge=0),
    ) -> ConcatenateIterator[str]:
        """Run a single prediction on the model"""
    
        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()
    
        image_data = load_image(str(image))
        image_tensor = self.image_processor.preprocess(image_data, return_tensors='pt')['pixel_values'].half().cuda()
    
        # loop start
    
        # just one turn, always prepend image token
        inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        conv.append_message(conv.roles[0], inp)

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, timeout=20.0)
    
        with torch.inference_mode():
            thread = Thread(target=self.model.generate, kwargs=dict(
                inputs=input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_tokens,
                streamer=streamer,
                use_cache=True))
            thread.start()
            # workaround: second-to-last token is always " "
            # but we want to keep it if it's not the second-to-last token
            prepend_space = False
            for new_text in streamer:
                if new_text == " ":
                    prepend_space = True
                    continue
                if new_text.endswith(stop_str):
                    new_text = new_text[:-len(stop_str)].strip()
                    prepend_space = False
                elif prepend_space:
                    new_text = " " + new_text
                    prepend_space = False
                if len(new_text):
                    yield new_text
            if prepend_space:
                yield " "
            thread.join()
    

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

# 새로운 배치 처리 함수들 (기존 코드와 분리)
import os
import glob
import json
import gc

def batch_predict_images():
    """qwen_test.py와 동일한 이미지들로 배치 처리하여 성능 비교"""
    
    print("MiniGemini MGM 배치 처리 시작...")
    
    # 메모리 최적화 설정
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    # 모델 로딩 (기존 Predictor 클래스와 동일한 방식)
    disable_torch_init()
    
    # MGM 모델의 올바른 경로 사용
    model_path = "./checkpoints/MGM-13B"  # 또는 실제 MGM 모델이 저장된 경로
    
    # MGM 모델이 없으면 다운로드하거나 기본 경로 사용
    if not os.path.exists(model_path):
        print(f"MGM 모델을 찾을 수 없습니다: {model_path}")
        print("MGM 모델을 먼저 다운로드해야 합니다.")
        return
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, 
        model_name="mgm-13b", 
        model_base=None, 
        load_8bit=False, 
        load_4bit=False
    )
    
    print("MGM 모델 로딩 완료!")
    
    # 이미지 폴더 경로 설정 (qwen_test.py와 동일)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # MGM 폴더에서 프로젝트 루트의 image 폴더로 이동
    image_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))))), "image")

    if not os.path.exists(image_folder):
        print(f"Error: {image_folder} 폴더가 존재하지 않습니다.")
        return

    jpg_files = glob.glob(os.path.join(image_folder, "*.jpg")) + glob.glob(os.path.join(image_folder, "*.JPG"))

    if not jpg_files:
        print(f"Error: {image_folder} 폴더에 jpg 파일이 없습니다.")
        return

    print(f"찾은 이미지 파일: {len(jpg_files)}개")

    # 결과 저장 리스트
    results = []
    
    # 대화 모드 설정
    conv_mode = "llava_v1"
    conv_template = conv_templates[conv_mode]
    
    # 각 이미지 처리
    for i, image_path in enumerate(jpg_files):
        print(f"\n처리 중: {os.path.basename(image_path)} ({i+1}/{len(jpg_files)})")
        
        try:
            # 메모리 정리
            torch.cuda.empty_cache()
            gc.collect()
            
            # 이미지 로드
            image_data = load_image(image_path)
            image_tensor = image_processor.preprocess(image_data, return_tensors='pt')['pixel_values'].half().cuda()
            
            # 이미지 정보 출력
            print(f"이미지 크기: {image_data.size}, 모드: {image_data.mode}")
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"CUDA 메모리 - 할당: {memory_allocated:.2f}GB, 예약: {memory_reserved:.2f}GB")
            
            # 대화 설정 (기존 predict 함수와 동일)
            conv = conv_template.copy()
            prompt = "이 이미지를 한 줄로 자세히 설명해주세요."
            inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt_text = conv.get_prompt()
            
            # 토크나이저 처리
            input_ids = tokenizer_image_token(prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            
            # 추론 실행
            with torch.inference_mode():
                output_ids = model.generate(
                    inputs=input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    max_new_tokens=128,
                    use_cache=True
                )
            
            # 출력 디코딩
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            
            # 프롬프트 부분 제거
            if conv.sep in outputs:
                outputs = outputs.split(conv.sep)[-1].strip()
            
            # 결과 저장
            result = {
                "image_index": i,
                "image_filename": os.path.basename(image_path),
                "image_path": image_path,
                "description": outputs if outputs else "No description generated"
            }
            results.append(result)
            
            print(f"설명: {outputs if outputs else 'No description generated'}")
            
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
    output_file = os.path.join(script_dir, "output_mgm.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n모든 이미지 처리 완료!")
    print(f"결과가 {output_file}에 저장되었습니다.")
    print(f"총 {len(results)}개의 이미지 처리됨")

# 배치 처리 실행을 위한 메인 함수
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        batch_predict_images()
    else:
        print("배치 처리를 원하면: python predict.py batch")
        print("기존 Predictor 클래스는 그대로 사용 가능합니다.")
