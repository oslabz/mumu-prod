from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse  # FileResponse 추가
from schemas import QueryModel
from file import embed_file
import os
import httpx
import zipfile
import logging
import pandas as pd # type: ignore
import json

router = APIRouter(
    prefix="/create",
    tags=["create"]
)

logger = logging.getLogger(__name__)

# 필수 디렉토리 생성
os.makedirs(".cache/embeddings", exist_ok=True)
os.makedirs(".cache/files", exist_ok=True)
os.makedirs(".cache/excel", exist_ok=True)
timeout = httpx.Timeout(120.0, read=120.0)
OLLAMA_URL = "http://20.39.201.16:11434/api/generate"

RAG_PROMPT_TEMPLATE1 = """당신은 질문에 EXCEL 형식의 배열로만 답변하는 AI입니다. 주어진 문맥을 사용하여 EXCEL 형식의 배열로 답변하세요. 답이 EXCEL형식이 아니면 대답을 하지마세요.
Question: {question}
Context: {context}
Answer:"""

# 전역 retriever 변수 선언
retriever = None

# 응답을 저장할 전역 변수
query_response_cache = {}

async def query_ollama(inputs: str) -> dict:
    try:
        payload = {
            "model": "EEVE-Korean-10.8B:latest",
            "prompt": inputs,
            "stream": False,
            "max_tokens": 10000
        }
        headers = {
            'Content-Type': 'application/json'
        }
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(OLLAMA_URL, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            logger.debug(f"OLLAMA response: {result}")  # 응답 로깅 추가

            # 응답이 JSON 형식인지 확인
            if 'response' in result:
                return result
            else:
                raise HTTPException(status_code=500, detail="Invalid response structure from OLLAMA")
    except httpx.HTTPStatusError as http_error:
        raise HTTPException(status_code=http_error.response.status_code, detail=http_error.response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def process_query(inputs: str) -> str:
    # 첫 번째 질의
    if retriever:
        context_docs = retriever.invoke(inputs)  # 동기적으로 retriever.invoke 호출
        context = "\n\n".join(doc.page_content for doc in context_docs)
    else:
        context = ""
    
    prompt1 = RAG_PROMPT_TEMPLATE1.format(question=inputs, context=context)
    result = await query_ollama(prompt1)
    
    answer1 = result['response']
    logger.info(f"Received answer1: {answer1}")  # answer1의 내용을 출력

    return answer1  

@router.post("/query")
async def query_doc(query: QueryModel):
    global query_response_cache
    try:
        answer = await process_query(query.inputs)
        query_response_cache = answer  # 응답을 전역 변수에 저장
        logger.debug(f"query_response_cache: {query_response_cache}")  # query_response_cache 로깅 추가
        return JSONResponse(content={"response": answer})
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload_directory")
async def upload_directory(file: UploadFile = File(...)):
    try:
        # 압축 파일 저장 경로 설정
        zip_path = f"./.cache/files/{file.filename}"
        if not zip_path.endswith(".zip"):
            raise HTTPException(status_code=400, detail="Uploaded file is not a zip file")

        with open(zip_path, "wb") as f:
            f.write(await file.read())  # 비동기 파일 읽기

        # 압축 해제
        dir_path = zip_path.replace(".zip", "")
        os.makedirs(dir_path, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dir_path)

        # 디렉토리가 비어있는지 확인
        if not os.listdir(dir_path):
            raise HTTPException(status_code=400, detail="압축 해제된 디렉토리가 비어 있습니다")

        global retriever
        retriever = embed_file(dir_path, "dir")

        # retriever가 제대로 초기화되었는지 확인
        if retriever is None:
            raise HTTPException(status_code=500, detail="파일 임베딩 중 오류가 발생했습니다")

        return {"status": "success", "message": "디렉토리의 파일 임베딩이 성공적으로 완료되었습니다"}
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid zip file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def response_to_excel(data_dict, output_file):
    try:
        # 데이터를 행과 열로 변환
        data = {}
        for item in data_dict:
            for key, value in item.items():
                col = key[:-1]  # 'A1'에서 'A'를 추출
                row = int(key[1:])  # 'A1'에서 '1'을 추출

                if col not in data:
                    data[col] = {}
                data[col][row] = value

        # 데이터프레임으로 변환
        df = pd.DataFrame(data).sort_index()

        # 엑셀 파일로 저장
        df.to_excel(output_file, index=False)
        print(f"Data successfully written to {output_file}")

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except KeyError as e:
        print(f"Key error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

@router.post("/excel")
async def json_to_excel():
    global query_response_cache
    try:
        if not query_response_cache:
            raise HTTPException(status_code=400, detail="No query response available to convert")

        # JSON 데이터를 파싱하여 딕셔너리로 변환
        try:
            data_dict = json.loads(query_response_cache)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Error decoding JSON: {e}")
        
        output_file = './response_output.xlsx'
        response_to_excel(data_dict, output_file)
        # 엑셀 파일을 반환하기 위해 FileResponse 사용
        return FileResponse(output_file, filename="response_output.xlsx", media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
