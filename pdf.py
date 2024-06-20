from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_community.vectorstores.faiss import FAISS
from schemas import QueryModel
import os
import httpx

router = APIRouter(
    prefix="/pdf",
    tags=["pdf"]
)

# 필수 디렉토리 생성
os.makedirs(".cache/embeddings", exist_ok=True)
os.makedirs(".cache/files", exist_ok=True)
timeout = httpx.Timeout(120.0, read=120.0)
OLLAMA_URL = "http://20.39.201.16:11434/api/generate"

RAG_PROMPT_TEMPLATE = """당신은 질문에 친절하게 답변하는 OSSLAB AI입니다. 주어진 문맥을 사용하여 질문에 답변하세요. 답을 모른다면 모른다고 답변하세요.
Question: {question}
Context: {context}
Answer:"""

#class QueryModel(BaseModel):
#    inputs: str

def embed_file(file_path):
    cache_dir = LocalFileStore(f"./.cache/embeddings/{os.path.basename(file_path)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""],
        length_function=len,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=text_splitter)

    model_name = "/home/osslab/mumul_dev/bge-m3"
    model_kwargs = {
        "device": "cpu"
    }
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, embedding=cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_path = f"./.cache/files/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        global retriever
        retriever = embed_file(file_path)
        return {"status": "success", "message": "파일 임베딩이 성공적으로 완료되었습니다"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 맞춤 HTTP 요청 단계를 정의
async def query_ollama(inputs: str) -> str:
    try:
        payload = {
            "model": "EEVE-Korean-10.8B:latest",
            "prompt": inputs,
            "stream": False
        }
        headers = {
            'Content-Type': 'application/json'
        }
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(OLLAMA_URL, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            return result['response']  # JSONResponse가 아니라 JSON 데이터 반환
    except httpx.HTTPStatusError as http_error:
        raise HTTPException(status_code=http_error.response.status_code, detail=http_error.response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 체인 정의
async def process_query(inputs: str) -> str:
    # Retrieve context
    context_docs = retriever.invoke(inputs)  # Synchronously call retriever.invoke
    context = "\n\n".join(doc.page_content for doc in context_docs)

    # Create prompt
    prompt = RAG_PROMPT_TEMPLATE.format(question=inputs, context=context)

    # Query Ollama
    answer = await query_ollama(prompt)

    return answer  # JSON 데이터 반환

@router.post("/query")
async def query_doc(query: QueryModel):
    try:
        answer = await process_query(query.inputs)
        return JSONResponse(content=answer)  # 여기서 JSONResponse로 감싸서 반환
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
