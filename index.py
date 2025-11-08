import os
import shutil
import asyncio
import time
import psutil
from pathlib import Path
from typing import List, Optional, Tuple
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from cachetools import LRUCache, TTLCache
from hashlib import blake2s
import re
import google.generativeai as genai
from pathlib import Path
from typing import List, Optional, Tuple

app = FastAPI(title="RAG + Gemini (Accuracy‑Maximized)", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
load_dotenv()
# --------- Paths & Persist ----------
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR = Path("./faiss_index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# --------- Embeddings & LLM ----------
embeddings: Optional[HuggingFaceEmbeddings] = None
gemini_client = None
GEMINI_MODEL = "gemini-2.5-flash"

# --------- Prompt -----------
SYSTEM_PROMPT = """
Bạn là trợ lý học vụ. Chỉ trả lời bằng tiếng Việt.

QUY TẮC:
1. Chỉ dùng thông tin trong “Ngữ cảnh”. Không dùng kiến thức ngoài.
2. Mọi số liệu hoặc kết luận phải kèm trích dẫn dạng [file:ĐiềuX].
3. Nếu thiếu dữ liệu hoặc không gán được nguồn, trả: "Không đủ dữ liệu."
4. Không suy diễn, không thêm nội dung.
5. Câu trả lời phải NGẮN GỌN: tối đa 3 đến 5 câu, chỉ nêu ý chính liên quan trực tiếp câu hỏi. Không liệt kê dài, không giải thích thừa.

Cuối trả lời ghi mục "Nguồn:" liệt kê các thẻ đã dùng.

Ngữ cảnh:
{context}
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}")
])

# --------- VectorStore & Retriever ----------
vectorstore: Optional[FAISS] = None

def load_vectorstore() -> Optional[FAISS]:
    if any(INDEX_DIR.iterdir()):
        try:
            return FAISS.load_local(
                str(INDEX_DIR),
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            print("Load FAISS failed:", e)
    return None

def persist_vectorstore(vs: FAISS) -> None:
    vs.save_local(str(INDEX_DIR))

# --------- Accuracy helpers ----------
def norm_text(s: str) -> str:
    return " ".join(s.strip().split()).lower()

def hash_ctx(question: str, context: str, model: str, k: int) -> str:
    h = blake2s(digest_size=16)
    h.update(norm_text(question).encode("utf-8"))
    h.update(b"|")
    h.update(context.encode("utf-8"))
    h.update(f"|{model}|{k}".encode("utf-8"))
    return h.hexdigest()

def _format_docs(docs: List[Document], per_chunk_limit: int = 2000, k_limit: int = 8) -> str:
    formatted_chunks = []
    
    for doc in docs[:k_limit]:
        meta = doc.metadata or {}
        
        src_name = Path(meta.get("source", "unknown")).name
        
        dieu_so = meta.get('dieu_so')
        
        if dieu_so is not None:
            source_tag = f":Điều{dieu_so}"
        else:
            source_tag = ":Không xác định" 
            
        tag = f"{src_name}{source_tag}"
        
        text = " ".join(doc.page_content.strip().split()) 
        
        if len(text) > per_chunk_limit:
            text = text[:per_chunk_limit] + "..."
            
        formatted_chunks.append(f"[{tag}] {text}")

    return "\n\n---\n\n".join(formatted_chunks)

def _make_retriever(k: int = 8):
    if vectorstore is None:
        raise RuntimeError("Vectorstore chưa sẵn sàng. Hãy gọi /ingest trước.")
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})

def _build_rag_chain(k: int = 8):
    retriever = _make_retriever(k)
    return (
        {
            "context": retriever | RunnableLambda(lambda docs: _format_docs(docs, per_chunk_limit=2000, k_limit=k)),
            "question": RunnablePassthrough()
        }
        | prompt
        | RunnableLambda(lambda inputs: _call_gemini(inputs["context"], inputs["question"]))
        | StrOutputParser()
    )

# --------- Gemini call helper -----------
def _call_gemini(context: str, question: str) -> str:
    prompt = f"""
    Bạn là trợ lý thông minh. Dưới đây là ngữ cảnh:
    {context}

    Câu hỏi: {question}
    """
    
    model = genai.GenerativeModel(
        "gemini-2.5-flash",
        generation_config={
            "temperature": 0.2,
            "top_p": 0.9,
            "max_output_tokens": 4096
        }
    ) 

    response = model.generate_content(prompt)
    # if not response.text:
    #     if response.prompt_feedback.block_reason != 0: 
    #         print(f"LỖI PHẢN HỒI BỊ CHẶN: Lý do chặn: {response.prompt_feedback.block_reason.name}")
    #         return "Lỗi: Yêu cầu bị hệ thống an toàn chặn. Vui lòng kiểm tra lại câu hỏi hoặc ngữ cảnh."
        

    # if response.candidates and response.candidates[0].finish_reason != 0:
    #     return f"Lỗi: Mô hình không thể hoàn thành phản hồi. Finish Reason: {response.candidates[0].finish_reason.name}"

    return response.text

# --------- Caching ---------
CONTEXT_CACHE = LRUCache(maxsize=50)
ANSWER_CACHE = TTLCache(maxsize=50, ttl=600)

async def rag_answer(question: str, k: int = 8) -> str:
    qn = norm_text(question)
    ctx_text = CONTEXT_CACHE.get((qn, k))

    if ctx_text is None:
        start_time_faiss = time.time()
        retriever = _make_retriever(k)
        docs = await asyncio.to_thread(retriever.get_relevant_documents, question)
        faiss_duration = time.time() - start_time_faiss
        print(f"[TIME] FAISS Retrieval: {faiss_duration:.4f} s")

        ctx_text = _format_docs(docs, per_chunk_limit=2000, k_limit=k)
        CONTEXT_CACHE[(qn, k)] = ctx_text
    else:
        print("[TIME] Context found in cache.")

    ck = hash_ctx(qn, ctx_text, model=GEMINI_MODEL, k=k)
    if ck in ANSWER_CACHE:
        return ANSWER_CACHE[ck]

    answer = await asyncio.to_thread(_call_gemini, ctx_text, question)
    ANSWER_CACHE[ck] = answer
    return answer

# --------- Ingest pipeline ----------
def _load_single_file_to_docs(path: Path) -> List[Document]:
    docs = []
    if path.suffix.lower() == ".docx":
        docs = Docx2txtLoader(str(path)).load()

    elif path.suffix.lower() == ".pdf":
        docs = PyPDFLoader(str(path)).load()

    elif path.suffix.lower() in [".txt", ".md"]:
        docs = TextLoader(str(path), encoding="utf-8").load()

    else:
        raise ValueError(f"Định dạng không hỗ trợ: {path.suffix}")

    for d in docs:
        d.metadata = d.metadata or {}
        d.metadata["source"] = str(path.resolve())

    return docs

def split_by_dieu(docs: List[Document]) -> List[Document]:
    new_docs = []
    
    pattern = r"(Điều\s*\d+\.?)"

    for doc in docs:
        content = doc.page_content.strip()

        ontent = re.sub(r"[ \t]+", " ", content)

        parts = re.split(pattern, content)

        for i in range(1, len(parts), 2):
            dieu_title = parts[i].strip()

            nums = re.findall(r"\d+", dieu_title)
            if not nums:
                continue

            dieu_number = int(nums[0])

            dieu_body = parts[i + 1].strip()

            if len(dieu_body) < 30:
                continue


            new_docs.append(Document(
                page_content=f"{dieu_title}\n{dieu_body}",
                metadata={
                    "source": doc.metadata.get("source", "unknown"),
                    "dieu_so": dieu_number
                }
            ))

    return new_docs


def _split_docs(docs: List[Document]) -> List[Document]:
    return split_by_dieu(docs)


def _index_documents(docs: List[Document]) -> None:
    global vectorstore
    chunks = _split_docs(docs)
    if vectorstore is None:
        vectorstore = FAISS.from_documents(chunks, embeddings)
    else:
        vectorstore.add_documents(chunks)
    persist_vectorstore(vectorstore)

def print_ram(label=""):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    print(f"[RAM] {label}: {mem_mb:.2f} MB")
# --------- FastAPI endpoints ----------
@app.on_event("startup")
def _startup():
    global embeddings, vectorstore

    print_ram("Before loading embeddings")

    if embeddings is None:
       embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )


    print_ram("After loading embeddings")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Chưa thiết lập GOOGLE_API_KEY")

    genai.configure(api_key=api_key)

    print_ram("Before loading FAISS")

    vs = load_vectorstore()
    if vs:
        vectorstore = vs
        print("FAISS index loaded.")
    else:
        print("No FAISS index yet. Upload at /ingest")

    print_ram("After loading FAISS")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": GEMINI_MODEL,
        "index_ready": vectorstore is not None,
        "answer_cache_items": len(ANSWER_CACHE),
        "context_cache_items": len(CONTEXT_CACHE)
    }

class AskRequest(BaseModel):
    question: str
    k: int = 8

@app.post("/ask")
async def ask(req: AskRequest):
    if req.k < 1 or req.k > 12:
        raise HTTPException(status_code=400, detail="k phải trong [1..12]")
    start_time = time.time() 
    answer = await rag_answer(req.question, k=req.k)
    elapsed = time.time() - start_time 
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    formatted_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    return JSONResponse(content={
        "answer": answer,
        "elapsed_seconds": round(elapsed, 3),
        "elapsed_hms": formatted_time 
    })

@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    saved_paths = []
    for f in files:
        dest = UPLOAD_DIR / f.filename
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        saved_paths.append(dest)

    all_docs = []
    for p in saved_paths:
        try:
            all_docs += _load_single_file_to_docs(p)
        except Exception as e:
            print(f"Skip {p.name}: {e}")

    if not all_docs:
        raise HTTPException(status_code=400, detail="Không có tài liệu hợp lệ để index.")

    _index_documents(all_docs)
    CONTEXT_CACHE.clear()
    ANSWER_CACHE.clear()
    return {
        "indexed_files": [Path(p).name for p in saved_paths],
        "total_docs": len(all_docs),
        "message": "Ingest OK. Index saved."
    }

@app.post("/reset_index")
def reset_index():
    global vectorstore
    for p in INDEX_DIR.glob("*"):
        if p.is_file():
            p.unlink()
        else:
            shutil.rmtree(p)
    vectorstore = None
    CONTEXT_CACHE.clear()
    ANSWER_CACHE.clear()
    return {"message": "Index removed. Re-ingest to rebuild."}

@app.get("/debug/faiss")
def debug_faiss(dump: bool = False):
    global vectorstore

    if not INDEX_DIR.exists() or not any(INDEX_DIR.iterdir()):
        return JSONResponse(
            {"error": "FAISS index not found or empty"},
            status_code=404
        )

    if vectorstore is None:
        vs = load_vectorstore()
        if not vs:
            return JSONResponse({"error": "Không thể load FAISS index"}, status_code=500)
        vectorstore = vs

    all_docs = list(vectorstore.docstore._dict.values())

    total_docs = len(all_docs)
    dieu_list = sorted([
        d.metadata.get("dieu_so")
        for d in all_docs
        if "dieu_so" in d.metadata
    ])

    preview = [
        {
            "dieu": d.metadata.get("dieu_so"),
            "text": d.page_content[:180].replace("\n", " ")
        }
        for d in all_docs
    ]

    result = {
        "total_docs": total_docs,
        "danh_sach_dieu": dieu_list,
        "preview": preview,
    }

    if dump:
        dump_path = INDEX_DIR / "debug_dump.txt"
        with dump_path.open("w", encoding="utf-8") as f:
            for d in all_docs:
                so = d.metadata.get("dieu_so", "N/A")
                f.write(f"=== Điều {so} ===\n")
                f.write(d.page_content)
                f.write("\n\n")
        result["dump_file"] = str(dump_path)

    return JSONResponse(result)