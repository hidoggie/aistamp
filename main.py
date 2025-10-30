# main.py (AI 비전 서버)

import os
import re
import io
import base64
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware # CORS 미들웨어 추가
from PIL import Image
import fitz
import google.generativeai as genai

# --- 설정 ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
# Render Secret Files 경로
GOOGLE_CREDENTIALS_PATH = "/etc/secrets/google-credentials.json"

# --- 전역 변수 ---
app: FastAPI
MODEL = None
INITIALIZATION_ERROR = None

# --- Lifespan (서버 시작 시 실행) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, REFERENCE_IMAGES, INITIALIZATION_ERROR
    print("✨ AI 비전 서버 리소스 초기화를 시작합니다...")
    try:
       REFERENCE_IMAGES = {}
       reference_dir = Path(__file__).resolve().parent / "reference_images"

        # reference_images 폴더를 순회하며 이미지들을 메모리에 로드
       for item_dir in reference_dir.iterdir():
            if item_dir.is_dir():
                item_name = item_dir.name # 예: 'a_sculpture'
                images = []
                for image_path in item_dir.glob("*.jpg"): # .jpg 등 다른 확장자도 가능
                    images.append(Image.open(image_path))
                REFERENCE_IMAGES[item_name] = images
        
            print(f"✅ 메모리에 {len(REFERENCE_IMAGES)}개 객체의 참조 이미지를 로드했습니다.")

        
        # Gemini 모델 초기화
        # 참고: 이 서버는 STT/TTS가 필요 없으므로 해당 클라이언트는 초기화하지 않음
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDENTIALS_PATH
            genai.configure(api_key=GEMINI_API_KEY)
            MODEL = genai.GenerativeModel('gemini-1.5-pro-flash')
            print("🎉 모든 리소스 초기화 완료.")
    except Exception as e:
        INITIALIZATION_ERROR = f"[{type(e).__name__}] {e}"
        print(f"💥 FATAL: 앱 초기화 중 오류 발생! 원인: {INITIALIZATION_ERROR}")
    yield
    print("👋 서버를 종료합니다.")

app = FastAPI(lifespan=lifespan)

# --- CORS 설정 ---
# 다른 도메인(suziestamp)에서 오는 요청을 허용하기 위해 필요
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 실제 운영 시에는 ['https://suziestamp.onrender.com'] 처럼 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ✨ 1. '폴더명'과 '행성 ID'를 연결하는 '번역 맵'을 추가합니다.
# 이 맵을 수정하여 각 객체와 스탬프 존을 연결합니다.
OBJECT_TO_PLANET_MAP = {
    "stamp_1": "earth",   
    "stamp_2": "mars",    
    "stamp_3": "jupiter",
    "stamp_4": "saturn",    
    "stamp_5": "neptune",
    # ... (인식할 모든 객체 폴더와 행성 ID를 짝지어 줍니다)
}

# --- API 엔드포인트 ---
@app.post("/api/recognize-stamp-object")
async def recognize_stamp_object(payload: dict = Body(...)):
    if INITIALIZATION_ERROR:
        raise HTTPException(status_code=500, detail=f"서버 초기화 오류: {INITIALIZATION_ERROR}")

    user_image_b64 = payload.get("image")
    if not user_image_b64:
        raise HTTPException(status_code=400, detail="이미지 데이터가 없습니다.")

    try:
        user_image_bytes = base64.b64decode(user_image_b64.split(',')[1])
        user_image = Image.open(io.BytesIO(user_image_bytes))

        # Gemini에게 이미지 매칭 요청
        prompt_parts = [
"You are an object recognition expert. The first image is from a user. Following that are sets of reference images, each set belonging to a single object.",
"Your task is to identify which object set the user's image belongs to.",
"**Important Instruction:** In all reference images, ignore the background, any tables, or surroundings. Focus ONLY on the primary object itself (e.g., the sculpture, the signboard) for matching.",
            # "중요 지시: 모든 참조 이미지에서, 배경, 탁자, 또는 주변 환경을 무시하십시오. 오직 핵심 객체(예: 조각상, 안내판) 자체에만 집중하여 일치 여부를 판단하십시오."
"Respond ONLY with the object's name in the format `[MATCH: object_name]`. If no match, respond with `[NO_MATCH]`.",
            user_image,
        ]
# 메모리에 로드된 모든 참조 이미지를 프롬프트에 추가
        for item_name, images in REFERENCE_IMAGES.items():
            prompt_parts.append(f"--- Reference Object: {item_name} ---")
            prompt_parts.extend(images)

        print("🤖 Gemini에게 이미지 매칭 요청...")
        gemini_response = await MODEL.generate_content_async(prompt_parts)
        match = re.search(r"\[MATCH:\s*(page_\d+_img_\d+\.jpg)\]", gemini_response.text)
        
        if match:
            matched_object_name = match.group(1).strip()
            return {"status": "success", "planet_id": matched_object_name} # 인식된 객체 이름 반환 
 
      #      planet_id = "sun" # TODO: 실제 규칙으로 변경 필요
      #      return {"status": "success", "planet_id": planet_id}
        else:
            print("❌ 일치하는 이미지를 찾지 못함.")
            return {"status": "no_match"}
    except Exception as e:
        print(f"💥 이미지 인식 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))