# main.py (이미지 벡터 검색 최종 버전)

import os, re, io, base64
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import google.generativeai as genai

# --- 1. 설정 ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GOOGLE_CREDENTIALS_PATH = "/etc/secrets/google-credentials.json"

# ✨ 1. 임베딩(지문 추출)과 응답 생성 모두 동일한 최신 모델을 사용합니다.
MODEL_NAME = "gemini-1.5-pro-flash"

# --- 2. 전역 변수 ---
app: FastAPI
MODEL: genai.GenerativeModel = None # 모델 객체
REFERENCE_VECTORS = [] # { 'id': 'a_sculpture', 'vector': [0.1, 0.2, ...] }
OBJECT_TO_PLANET_MAP = {
    "stamp_1": "earth",   
    "stamp_2": "mars",    
    "stamp_3": "jupiter",
    "stamp_4": "saturn",    
    "stamp_5": "neptune",
}
INITIALIZATION_ERROR = None

# --- 3. Lifespan - 이미지 지문(벡터) 생성 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, REFERENCE_VECTORS, INITIALIZATION_ERROR
    print("✨ AI 비전 서버 리소스 초기화를 시작합니다...")
    try:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDENTIALS_PATH
        genai.configure(api_key=GEMINI_API_KEY)
        
        # ✨ 2. 모델을 한 번만 초기화합니다.
        MODEL = genai.GenerativeModel(MODEL_NAME)
        
        reference_dir = Path(__file__).resolve().parent / "reference_images"
        print("🤖 참조 이미지들의 임베딩 벡터를 생성합니다...")
        
        for item_dir in reference_dir.iterdir():
            if item_dir.is_dir():
                item_name = item_dir.name
                for image_path in item_dir.glob("*.png"): # .png, .jpg 등 확장자 지원
                    print(f"  - {item_name} / {image_path.name} 지문 추출 중...")
                    img = Image.open(image_path)
                    # ✨ 3. AI에게 이미지 1장을 보내 '이미지 벡터'를 요청합니다.
                    response = MODEL.embed_content(
                        content=img, 
                        task_type="RETRIEVAL_DOCUMENT" # "이것은 DB용 문서입니다"
                    )
                    REFERENCE_VECTORS.append({
                        "id": item_name,
                        "vector": response['embedding']
                    })
        print(f"✅ {len(REFERENCE_VECTORS)}개의 참조 벡터를 메모리에 로드했습니다.")
    except Exception as e:
        INITIALIZATION_ERROR = f"[{type(e).__name__}] {e}"
        print(f"💥 FATAL: 앱 초기화 중 오류 발생! 원인: {INITIALIZATION_ERROR}")
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- 4. 벡터 유사도 계산 함수 ---
def find_best_match(query_vector):
    best_match_id = None
    best_score = -1
    
    query_vec = np.array(query_vector)
    
    for item in REFERENCE_VECTORS:
        item_vec = np.array(item["vector"])
        # 코사인 유사도 계산
        similarity = np.dot(query_vec, item_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(item_vec))
        
        if similarity > best_score:
            best_score = similarity
            best_match_id = item["id"]
            
    print(f"가장 유사한 객체: {best_match_id} (유사도: {best_score:.2f})")
    # ✨ 유사도 점수가 0.7 이상일 때만 일치로 인정 (이 값은 조절 가능)
    return best_match_id if best_score > 0.7 else None

# --- 5. API 엔드포인트 ---
@app.post("/api/recognize-stamp-object")
async def recognize_stamp_object(payload: dict = Body(...)):
    if INITIALIZATION_ERROR: raise HTTPException(status_code=500, detail=f"{INITIALIZATION_ERROR}")

    user_image_b64 = payload.get("image")
    if not user_image_b64: raise HTTPException(status_code=400, detail="이미지 데이터가 없습니다.")

    try:
        user_image_bytes = base64.b64decode(user_image_b64.split(',')[1])
        user_image = Image.open(io.BytesIO(user_image_bytes))

        # ✨ 4. 사용자 이미지의 '이미지 벡터' 추출
        print("🤖 사용자 이미지의 벡터 추출 요청...")
        response = MODEL.embed_content(
            content=user_image, 
            task_type="RETRIEVAL_QUERY" # "이것은 검색용 질문입니다"
        )
        query_vector = response['embedding']
        
        # 5. 저장된 참조 벡터들과 고속 비교
        matched_object_name = find_best_match(query_vector)
        
        if matched_object_name:
            planet_id = OBJECT_TO_PLANET_MAP.get(matched_object_name)
            if planet_id:
                return {"status": "success", "planet_id": planet_id}
            else:
                return {"status": "no_match", "description": "매칭 오류: 객체에 연결된 행성이 없습니다."}
        else:
            return {"status": "no_match", "description": "일치하는 전시물을 찾을 수 없습니다."}
            
    except Exception as e:
        print(f"💥 이미지 인식 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))