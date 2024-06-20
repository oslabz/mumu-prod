from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import httpx
from schemas import QueryModel
import logging

router = APIRouter(
    prefix="/answer",
    tags=["answer"]
)

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://20.39.201.16:11434/api/generate"
timeout = httpx.Timeout(120.0, read=120.0)

# PROMPT_TEMPLATE = """당신은 질문에 친절하게 답변하는 MUMULBOT AI입니다. 질문에 답변하세요. 답을 모른다면 모른다고 답변하세요.
# Question: {question}
# Answer:"""

PROMPT_TEMPLATE = """당신은 질문에 친절하게 답변하는 MUMULBOT AI입니다. 질문에 답변하세요. 답을 모른다면 모른다고 답변하세요.
카세트 번호 분류 및 검증
Description:

이 프롬프트는 카세트 번호 형식을 검증하고, 짧은 카세트와 긴 카세트로 분류하는 방법을 설명합니다. 카세트 번호는 총 6자리이며, 앞 2자리는 "DR" 또는 "IR"로 시작하고, 뒤 4자리는 숫자로 구성됩니다. 4번째 자리가 0으로 시작하는 카세트는 짧은 카세트로, 0이 아닌 숫자로 시작하는 카세트는 긴 카세트로 분류됩니다. 또한, 분류된 카세트를 좌우 균형 맞추기, 적재 행의 배치, 항구 순서 및 하역 순서를 고려하여 배치합니다.
Validation Rules:

카세트 번호는 총 6자리여야 합니다.
앞 2자리는 "DR" 또는 "IR"로 시작해야 합니다.
뒤 4자리는 숫자로 구성되어야 합니다.
4번째 자리가 0이면 짧은 카세트로 분류합니다.
4번째 자리가 0이 아니면 긴 카세트로 분류합니다.
Example Input

cassette_numbers "DR004", "DR005", "DR021", "DR032", "IR008", "IR014", "DR521", "DR531", "DR536", "DR555", "IR513", "IR519", "IR522", "IR530", "IR539", "IR541", "IR703", "IR713"
Example Output

Short cassettes "DR004", "DR005", "DR021", "DR032", "IR008", "IR014"
Long cassettes "DR521", "DR531", "DR536", "DR555", "IR513", "IR519", "IR522", "IR530", "IR539", "IR541", "IR703", "IR713"

Additional Instructions:

좌우 균형 맞추기:

각 측면에 동일한 수의 긴 카세트와 짧은 카세트를 배치합니다.
왼쪽 측면: 긴 카세트 6개, 짧은 카세트 3개
오른쪽 측면: 긴 카세트 6개, 짧은 카세트 3개
적재 행의 배치:

각 적재 행에는 최대 4개의 긴 카세트를 배치하거나, 2개의 긴 카세트와 3개의 짧은 카세트를 배치할 수 있습니다.
한 행에 최대 5개의 카세트를 배치합니다.
짧은 카세트가 배치된 후에는 다음 행에 짧은 카세트가 배치될 수 없습니다.
3개의 짧은 카세트는 같은 열에 연달아 배치되어야 합니다.
항구 순서 및 하역 순서 고려:

항구 순서에 따라 하역이 쉽게 이루어질 수 있도록 배치합니다.
항구 순서 1 (마산): 먼저 하역
항구 순서 2 (거제): 다음 하역
하역항이 1개만 있는 경우, 항구 순서를 고려하지 않아도 됩니다.
최종적으로 배치 항구1, 2는 한 번에 실립니다.
Example Output

좌측1열 (좌측1): ["DR004", "DR005", "DR021", "IR530", "DR555"]
좌측2열 (좌측2): ["IR539", "IR703", "DR531", "IR522"]
우측3열 (우측3): ["DR521", "IR713", "DR536", "IR513"]
우측4열 (우측4): ["IR008", "IR014", "DR032", "IR541", "IR519"]
형식에 맞도록 답변해주세요.
Question: {question}
Answer:"""

async def query_ollama(inputs: str) -> dict:
    try:
        payload = {
            "model": "EEVE-Korean-10.8B:latest",
            "prompt": inputs,
            "stream": False #,
            #"max_tokens": 10000
        }
        headers = {
            'Content-Type': 'application/json'
        }
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(OLLAMA_URL, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            return result['response']
    except httpx.HTTPStatusError as http_error:
        raise HTTPException(status_code=http_error.response.status_code, detail=http_error.response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def process_query(inputs: str) -> str:
    prompt = PROMPT_TEMPLATE.format(question=inputs)
    answer = await query_ollama(prompt)
    return answer

@router.post("/query")
async def query_doc(query: QueryModel):
    try:
        answer = await process_query(query.inputs)
        logger.info(f"Received answer: {answer}")
        return JSONResponse(content={"answer": answer})
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
