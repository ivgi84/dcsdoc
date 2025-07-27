# qa_app.py

import pickle
import faiss
import torch
from langchain.schema import Document
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

# ————————————————————————————————————
# 1) Load FAISS index + text chunks
# ————————————————————————————————————
index = faiss.read_index("faiss.index")
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# ————————————————————————————————————
# 2) Vector store & retriever
# ————————————————————————————————————
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
docs = [Document(page_content=chunk) for chunk in chunks]
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ————————————————————————————————————
# 3) GPU‑accelerated FLAN‑T5 pipeline (Accelerate + FP16)
# ————————————————————————————————————
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-large",
    device_map="auto",            # Accelerate shards across GPU(s)
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

hf_pipe = Text2TextGenerationPipeline(
    model=model,
    tokenizer=tokenizer,
    max_length=256,
    min_length=50,
    do_sample=False,      # deterministic beam search
    num_beams=4,
    early_stopping=True,
    truncation=True,
)

# Wrap the HF pipeline in a LangChain‑compatible Runnable
llm = HuggingFacePipeline(pipeline=hf_pipe)   # ← key fix

# ————————————————————————————————————
# 4) Prompts for refine chain (document var = {context})
# ————————————————————————————————————
initial_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are an expert F‑16 flight instructor. Refer ONLY to the excerpt:\n\n"
        "{context}\n\n"
        "Provide a clear, numbered, step‑by‑step procedure for:\n"
        "{question}\n"
        "If the excerpt lacks relevant info, respond “INFO_NOT_FOUND.”"
    )
)

refine_prompt = PromptTemplate(
    input_variables=["existing_answer", "context", "question"],
    template=(
        "You are improving an F‑16 procedure.\n"
        "Current answer:\n{existing_answer}\n\n"
        "New excerpt:\n{context}\n\n"
        "Update the answer so it fully addresses:\n{question}\n"
        "- Keep it concise and numbered.\n"
        "- If the excerpt adds nothing, keep the answer unchanged."
    )
)

# ————————————————————————————————————
# 5) Build the refine QA chain
# ————————————————————————————————————
qa_chain = load_qa_chain(
    llm=llm,
    chain_type="refine",
    question_prompt=initial_prompt,
    refine_prompt=refine_prompt,
    document_variable_name="context",
)

# ————————————————————————————————————
# 6) Public function for Streamlit
# ————————————————————————————————————
def answer(question: str) -> str:
    """
    Retrieve top‑k relevant chunks and run the refine chain.
    """
    if not question:
        return "Please provide a question."
    docs = retriever.get_relevant_documents(question)
    return qa_chain.run(input_documents=docs, question=question)
