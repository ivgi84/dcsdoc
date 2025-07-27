# qa_app.py
"""
Streamlit‑ready QA helper:
    • loads the persisted FAISS store from ./faiss_store
    • wraps a GPU‑aware FLAN‑T5‑large pipeline (no unused kwargs)
    • answers questions with a Refine QA chain
"""
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Text2TextGenerationPipeline,
)

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
STORE_DIR = "faiss_store"           # folder created by indexer.py
K = 5                                # top‑k chunks to retrieve

# ─────────── 1) Load the vector‑store ───────────
embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)
vectorstore = FAISS.load_local(
    STORE_DIR,
    embeddings,
    allow_dangerous_deserialization=True,   # LangChain ≥0.2 safety switch
)
retriever = vectorstore.as_retriever(search_kwargs={"k": K})

# ─────────── 2) FLAN‑T5‑large pipeline (FP16) ───────────
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-large",
    device_map="auto",          # Accelerate chooses the GPU(s)
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
hf_pipe = Text2TextGenerationPipeline(
    model=model,
    tokenizer=tokenizer,
    max_length=256,
    min_length=50,
    do_sample=False,
    num_beams=4,
    early_stopping=True,
)

hf_pipe.task = "text2text-generation"      # ← add this line
llm = HuggingFacePipeline(pipeline=hf_pipe)


# ─────────── 3) Refine chain prompts ───────────
initial_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are an expert F‑16 flight instructor. Refer ONLY to the excerpt below.\n\n"
        "{context}\n\n"
        "Provide a clear, numbered, step‑by‑step procedure for:\n"
        "{question}\n\n"
        'If the excerpt lacks relevant info, respond exactly with "INFO_NOT_FOUND".'
    ),
)
refine_prompt = PromptTemplate(
    input_variables=["existing_answer", "context", "question"],
    template=(
        "You are improving an F‑16 procedure.\n\n"
        "Current answer:\n{existing_answer}\n\n"
        "New excerpt:\n{context}\n\n"
        "Update the answer so it fully addresses the question:\n{question}\n"
        "- Keep it concise and numbered.\n"
        "- If the new excerpt adds nothing, keep the answer unchanged."
    ),
)
qa_chain = load_qa_chain(
    llm=llm,
    chain_type="refine",
    question_prompt=initial_prompt,
    refine_prompt=refine_prompt,
    document_variable_name="context",
)

# ─────────── 4) Public helper for Streamlit ───────────
def answer(question: str) -> str:
    """Retrieve top‑k chunks and run the Refine QA chain."""
    question = (question or "").strip()
    if not question:
        return "Please enter a question."
    docs = retriever.get_relevant_documents(question)
    return qa_chain.run(input_documents=docs, question=question)
