import shutil
import asyncio
import time
from pathlib import Path
from typing import List, Optional, Tuple
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# LangChain
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

# Cache
from cachetools import LRUCache, TTLCache
from hashlib import blake2s

app = FastAPI(title="RAG + Ollama (Accuracy-Maximized)", version="1.2.0")

# --------- Paths & Persist ----------
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR = Path("./faiss_index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# --------- Embeddings & LLM ----------
# Dùng model embedding chuẩn & chunk đủ lớn để tránh mất ngữ nghĩa
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

llm = Ollama(
    model="gemma2:2b",
    temperature=0.1,         # giảm randomness xuống tối đa
    num_ctx=4096,            # tăng context window hết mức model cho phép
    num_predict=384,         # cho phép trả lời rất chi tiết lên tới nhiều nội dung hơn
    mirostat=0,
    repeat_penalty=1.3,      # tăng mạnh penalty để giảm lặp/ảo
    keep_alive="30m"
)

# --------- Prompt cực nghiêm ngặt -----------
SYSTEM_PROMPT = """Bạn là trợ lý khoa học chính xác tuyệt đối, chỉ được trả lời dựa trên NGỮ CẢNH dưới đây.
Rất quan trọng: MỌI LUẬN ĐIỂM, SỐ LIỆU, KẾT LUẬN phải có đối chiếu NGUỒN cụ thể từng ý, theo định dạng [file:pX].
Không bao giờ bịa đặt/tưởng tượng/chuyển diễn giải nếu thiếu nguồn. Nếu thông tin NGỮ CẢNH không đủ, chỉ trả lời: "không đủ dữ liệu".
Không sử dụng bất kỳ kiến thức ngoài nào – CHỈ trả lời dựa vào thông tin trong 'Ngữ cảnh' bên dưới.
Cuối câu trả lời, liệt kê danh sách 'Nguồn:' gồm các [file:pX] đã sử dụng.

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

def _format_docs(docs: List[Document], per_chunk_limit=2000, k_limit=8) -> str:
    # Lấy tối đa k_limit chunk, nhưng chunk rất to để không vỡ ý
    out = []
    n_chunks = min(len(docs), k_limit)
    for d in docs[:n_chunks]:
        meta = d.metadata or {}
        src = Path(meta.get("source", "unknown")).name
        page = meta.get("page", None)
        tag = f"{src}" + (f":p{page+1}" if page is not None else "")
        text = d.page_content.strip().replace("\n", " ")
        if len(text) > per_chunk_limit:
            text = text[:per_chunk_limit] + "..."
        out.append(f"[{tag}] {text}")
    return "\n\n---\n\n".join(out)

def _make_retriever(k: int = 8):
    if vectorstore is None:
        raise RuntimeError("Vectorstore chưa sẵn sàng. Hãy gọi /ingest trước.")
    # k lớn hết cỡ để quét nhiều tài liệu, tăng recall tối đa
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})

def _build_rag_chain(k: int = 8):
    retriever = _make_retriever(k)
    return (
        {
            "context": retriever | RunnableLambda(lambda docs: _format_docs(docs, per_chunk_limit=2000, k_limit=k)),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

# --------- Caching ---------
ANSWER_CACHE = TTLCache(maxsize=512, ttl=1800)  # giảm cache TTL để không trả lời sai do cache cũ
CONTEXT_CACHE = LRUCache(maxsize=1024)

async def rag_answer(question: str, k: int = 8) -> str:
    chain = _build_rag_chain(k=k)

    qn = norm_text(question)
    ctx_text = CONTEXT_CACHE.get((qn, k))
    if ctx_text is None:
        retriever = _make_retriever(k)
        docs = retriever.get_relevant_documents(question)
        ctx_text = _format_docs(docs, per_chunk_limit=2000, k_limit=k)
        CONTEXT_CACHE[(qn, k)] = ctx_text

    ck = hash_ctx(qn, ctx_text, model="gemma2:2b", k=k)
    if ck in ANSWER_CACHE:
        return ANSWER_CACHE[ck]

    loop = asyncio.get_event_loop()

    def run_llm():
        base = prompt | llm | StrOutputParser()
        return base.invoke({"context": ctx_text, "question": question})

    try:
        answer = await loop.run_in_executor(None, run_llm)
        ANSWER_CACHE[ck] = answer
        return answer
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM/RAG error: {e}")


# --------- Ingest pipeline ----------
def _load_single_file_to_docs(path: Path) -> List[Document]:
    docs = []
    if path.suffix.lower() == ".pdf":
        docs += PyPDFLoader(str(path)).load()
    elif path.suffix.lower() == ".docx":
        docs += Docx2txtLoader(str(path)).load()
    elif path.suffix.lower() in [".txt", ".md"]:
        docs += TextLoader(str(path), encoding="utf-8").load()
    else:
        raise ValueError(f"Định dạng không hỗ trợ: {path.suffix}")
    for d in docs:
        d.metadata = d.metadata or {}
        d.metadata["source"] = str(path.resolve())
    return docs

def _split_docs(docs: List[Document]) -> List[Document]:
    # Chunk cực lớn để tránh chia nhỏ thông tin context/nội dung câu trả lời
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=256,
        add_start_index=True
    )
    return splitter.split_documents(docs)

def _index_documents(docs: List[Document]) -> None:
    global vectorstore
    chunks = _split_docs(docs)
    if vectorstore is None:
        vectorstore = FAISS.from_documents(chunks, embeddings)
    else:
        vectorstore.add_documents(chunks)
    persist_vectorstore(vectorstore)

# --------- FastAPI endpoints ----------
@app.on_event("startup")
def _startup():
    global vectorstore
    vs = load_vectorstore()
    if vs:
        print("FAISS index loaded."); vectorstore = vs
    else:
        print("No FAISS index yet. Upload at /ingest")
    # Warmup model: gọi 1 prompt rất ngắn để load vào RAM
    try:
        _ = llm.invoke("Xin chào, kiểm tra khởi động.")
        print("Model warmup done.")
    except Exception as e:
        print("Warmup failed:", e)

@app.get("/health")
def health():
    return {"status": "ok", "model": "gemma2:2b", "index_ready": vectorstore is not None,
            "answer_cache_items": len(ANSWER_CACHE), "context_cache_items": len(CONTEXT_CACHE)}

class AskRequest(BaseModel):
    question: str
    k: int = 8  # k mặc định lớn nhất để tăng recall và độ phủ context

@app.post("/ask")
async def ask(req: AskRequest):
    if req.k < 1 or req.k > 12:
        raise HTTPException(status_code=400, detail="k phải trong [1..12]")
    answer = await rag_answer(req.question, k=req.k)
    return JSONResponse(content={"answer": answer})

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
    return {"indexed_files": [Path(p).name for p in saved_paths],
            "total_docs": len(all_docs), "message": "Ingest OK. Index saved."}

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
