from fastapi import APIRouter, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_community.document_loaders import TextLoader, UnstructuredHTMLLoader, DirectoryLoader, Docx2txtLoader, UnstructuredExcelLoader
from langchain_community.vectorstores.faiss import FAISS
from schemas import QueryModel
import os
import httpx
import json  # json 모듈을 임포트
import zipfile
import logging

router = APIRouter(
    prefix="/files",
    tags=["files"]
)

logger = logging.getLogger(__name__)

# 필수 디렉토리 생성
os.makedirs(".cache/embeddings", exist_ok=True)
os.makedirs(".cache/files", exist_ok=True)
timeout = httpx.Timeout(120.0, read=120.0)
OLLAMA_URL = "http://20.39.201.16:11434/api/generate"

RAG_PROMPT_TEMPLATE = """당신은 질문에 친절하게 답변하는 OSSLAB AI입니다. 주어진 문맥을 사용하여 질문에 답변하세요. 답을 모른다면 모른다고 답변하세요.
Question: {question}
Context: {context}
Answer:"""

retriever = None  # 전역 retriever 변수 선언

def embed_file(file_path, file_type):
    cache_dir = LocalFileStore(f"./.cache/embeddings/{os.path.basename(file_path)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""],
        length_function=len,
    )
    loaders = []
    if file_type == "pdf":
        loaders.append(UnstructuredFileLoader(file_path))
    elif file_type == "txt":
        loaders.append(TextLoader(file_path))
    elif file_type in ["doc", "docx"]:
        loaders.append(Docx2txtLoader(file_path))
    elif file_type in ["xls", "xlsx"]:
        loaders.append(UnstructuredExcelLoader(file_path))
    elif file_type == "html":
        loaders.append(UnstructuredHTMLLoader(file_path))
    elif file_type == "dir":
        loaders.extend([
            DirectoryLoader(file_path, glob="**/*.pdf", loader_cls=UnstructuredFileLoader),
            DirectoryLoader(file_path, glob="**/*.txt", loader_cls=UnstructuredFileLoader),
            DirectoryLoader(file_path, glob="**/*.doc", loader_cls=UnstructuredFileLoader),
            DirectoryLoader(file_path, glob="**/*.docx", loader_cls=UnstructuredFileLoader),
            DirectoryLoader(file_path, glob="**/*.xls", loader_cls=UnstructuredFileLoader),
            DirectoryLoader(file_path, glob="**/*.xlsx", loader_cls=UnstructuredFileLoader)
        ])
    else:
        raise ValueError("Unsupported file type")

    docs = []
    for loader in loaders:
        docs.extend(loader.load_and_split(text_splitter=text_splitter))

    if not docs:
        raise ValueError("No documents loaded and split")

    #docs = loader.load_and_split(text_splitter=text_splitter)
    logger.info(f"Loaded {len(docs)} documents for embedding")


    model_name = "/home/osslab/mumul_dev/bge-m3"
    model_kwargs = {"device": "cpu"}
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

async def query_ollama(inputs: str) -> str:
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
            return result['response']  # JSONResponse가 아니라 JSON 데이터 반환
    except httpx.HTTPStatusError as http_error:
        raise HTTPException(status_code=http_error.response.status_code, detail=http_error.response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def process_query(inputs: str) -> str:
    context_docs = retriever.invoke(inputs)  # Synchronously call retriever.invoke
    context = "\n\n".join(doc.page_content for doc in context_docs)

    prompt = RAG_PROMPT_TEMPLATE.format(question=inputs, context=context)
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

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_path = f"./.cache/files/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(file.file.read())

        file_extension = file.filename.split(".")[-1]
        if file_extension not in ["pdf", "txt", "doc", "docx", "xls", "xlsx", "json", "html"]:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        global retriever
        retriever = embed_file(file_path, file_extension)
        return {"status": "success", "message": "파일 임베딩이 성공적으로 완료되었습니다"}
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