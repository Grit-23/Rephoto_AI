import os
import json
from dotenv import load_dotenv
from PIL import Image
import base64

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

# .env 파일 로드 및 API 키 불러오기
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 이미지 폴더 경로
image_folder = "/home/yongbin53/Research/GalleryProj/image"
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

# GPT-4o 모델 초기화
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model_name="gpt-4.1-mini", max_tokens=500)
output_parser = StrOutputParser()

# 결과 저장 리스트
results = []

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # base64 인코딩
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    # GPT-4o는 vision input을 base64로 받음
    message = HumanMessage(
        content=[
            {"type": "text", "text": "이 이미지를 한 줄로 자세히 설명해주세요."},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}",
                    "detail": "high"
                }
            }
        ]
    )

    try:
        response = llm.invoke([message])
        description = response.content.strip()
    except Exception as e:
        description = f"Error: {str(e)}"

    results.append({
        "imgname": image_file,
        "description": description
    })

    print(f"Progress: {len(results)}/{len(image_files)} images processed")

# JSON 파일로 저장
with open("gpt/output_3.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)